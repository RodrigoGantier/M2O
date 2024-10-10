import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader
from accelerate import Accelerator

import numpy as np
import os
import argparse
from tqdm import tqdm, trange
from math import floor, ceil
import random
import numbers
 
from utils_.loss_fn import lpips, tcLoss, flowFn, metricsFn, Discriminator
from utils_.noisyDataLoader import noisyDataloader
from utils_.valDataLoader import valDataLoad
from models_.e2vid import UConvLSTM


def seed_everything(seed):
    
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class padd_fn:
    def __init__(self, img_size):
        self.w = img_size[1]
        self.h = img_size[0]
        self.net_scale = 8

        mod = (self.h % self.net_scale) / self.net_scale
        h_size = int((1 - mod) * self.net_scale * (mod>0))
        h_size += self.h

        mod = (self.w % self.net_scale) / self.net_scale
        w_size = int((1 - mod) * self.net_scale * (mod>0))
        w_size += self.w

        self.padding_top = ceil(0.5 * (h_size - self.h))
        self.padding_bottom = floor(0.5 * (h_size - self.h))

        self.padding_left = ceil(0.5 * (w_size - self.w))
        self.padding_right = floor(0.5 * (w_size - self.w))
        
    def padding(self, x):
        return F.pad(x, 
                     (self.padding_left, 
                      self.padding_right,
                      self.padding_top, 
                      self.padding_bottom,))
    
    def unpadd(self, x):
        if self.padding_left > 0:
            x = x[:, :, :, self.padding_left:]
        if self.padding_right > 0:
            x = x[:, :, :, :-self.padding_right]
        if self.padding_top > 0:
            x = x[:, :, self.padding_top:]
        if self.padding_bottom > 0:
            x = x[:, :, :-self.padding_bottom]

        return x


class Trainer:
    def __init__(self, args):

        self.debug = args.debug
        self.verbose = args.verbose
        if self.debug:
            args.num_workers = 0
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device
        # hyper parametes
        self.init_discr = args.init_discr
        self.tc_lambda = args.tc_lambda
        self.ss_lambda = args.ss_lambda
        self.adv_lambda = args.adv_lambda
        self.rec_lambda = args.rec_lambda
        self.epochs = args.epochs
        self.valid_freq = args.valid_freq # frequency tun testset
        self.adv_freq = args.adv_freq     # frequency train discriminator
        self.writer = SummaryWriter(args.ckpt_path)
        crop_size = args.transforms['randomCrop']['size']
        self.ckpt_path = args.ckpt_path
        # we assume that in training, all inputs (size) are the same size
        self.crop_size = (int(crop_size), int(crop_size)) if isinstance(crop_size, numbers.Number) else crop_size
        
        # load networks
        self.model = UConvLSTM(args).to(self.device)        # model
        self.lpips = lpips(model='alex').to(self.device)    # lpips loss (alex net)
        lpips_net = lpips(model='vgg').to(self.device)      # lpips metric (vgg net)
        flow_net = flowFn().to(self.device)                 # flow metric (RAFT)
        self.discr = Discriminator(1).to(self.device)       # adversarial loss
        
        # loss function
        # temporal consistency loss
        self.tc_loss = tcLoss() 
        self.tc_loss.set_params(self.crop_size) 
        # reconstruction L2 loss
        self.rec_loss = torch.nn.MSELoss()
        # metric for the validation stage
        self.metrics = metricsFn(lpips_net, flow_net)
        # discriminator loss
        self.adv_loss = torch.nn.BCEWithLogitsLoss()

        # load datasets
        train_dataset = noisyDataloader(args) 
        # activate persistent_workers if there is more than one worker
        persist_wkr = args.num_workers > 0 
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=args.num_workers, 
            persistent_workers=persist_wkr, 
            pin_memory=True, 
            drop_last=True)
        self.len_epoch = len(self.train_dataloader)  # for metrics 

        val_dataset = valDataLoad(args)
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False,
            )

        # setup optimezer
        self.optm = torch.optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            amsgrad=True,
            weight_decay=0,
            foreach=False)
        self.optm_d = torch.optim.Adam(
            self.discr.parameters(), 
            lr=args.discr_lr, betas=(0.5, 0.999))

        # setup scheduler
        self.scheduler = StepLR(
            self.optm, 
            step_size=50,
            gamma=0.9)
        self.sche_max = args.sche_max

        self.best_lpips = 1000
        self.start_epoch = 0
        self.mult = args.dl_mult

        # reload weghts
        if os.path.isdir(self.ckpt_path):
            self.resume_checkpoint()
        
        self.model, \
        self.optm, \
        self.train_dataloader, \
        self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optm, 
            self.train_dataloader, 
            self.scheduler)

    def resume_checkpoint(self, ):

        save_files = [self.ckpt_path +'/'+ d.name for d in os.scandir(self.ckpt_path) if d.name.endswith('pth')]
        save_dic = [d for d in save_files if 'disc-epoch' in d]
        save_files = [d for d in save_files if 'checkpoint-epoch' in d]
        
        save_files.sort()
        if len(save_files) < 1: 
            return None

        resume_path = save_files[-1] # retive the last ckpt

        print(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.best_lpips = checkpoint['best_lpips']

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optm.load_state_dict(checkpoint['optimizer'])
        self.optm_to_device(self.optm, self.device)  
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if type(self.model).__name__ == checkpoint['arch']: 
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
        #load discriminator weights
        if len(save_dic) > 1:
            save_dic.sort()
            discr_ckpt = save_dic[-1]
            discr_ckpt = torch.load(discr_ckpt, map_location='cpu')
            self.discr.load_state_dict(discr_ckpt['state_dict'], strict=True)

            if "optimizer" in discr_ckpt:
                self.optm_d.load_state_dict(discr_ckpt['optimizer'])
                self.optm_to_device(self.optm_d, self.device)
        

    @staticmethod
    def optm_to_device(optm, device):

        for param in optm.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                             subparam._grad.data = subparam._grad.data.to(device)
    
    def reset_state(self, ):

        if self.debug:
            for j in range(self.model.num_encoders):
                self.model.encoders[j].recurrent_block.state = None
        else:
            for j in range(self.model.module.num_encoders):
                self.model.module.encoders[j].recurrent_block.state = None

        
    
    def train_sequ(self, sequence, epoch):
        
        seq_len = sequence['events'].shape[1]
        self.reset_state()
        
        # the last image is not used in the loop
        for i in range(seq_len - 1):
            
            voxel = sequence['events'][:, i]
            pred = self.model(voxel)
        
        image_0 = sequence['img'][:, -2]  # t0 image
        image_1 = sequence['img'][:, -1]  # t1 image
        flow = sequence['flow'][:, -1]    # flow between t0 and t1 is used
        
        tc = self.tc_loss(pred, image_0, image_1, flow).to(flow.device)
        lpips = self.lpips(pred, image_0)
        rec_loss = self.rec_loss(pred, image_0)

        total_loss = lpips + (self.tc_lambda * tc) + (self.rec_lambda * rec_loss)
        if epoch > self.init_discr:
            fake = self.discr(pred)
            real = torch.ones_like(fake)
            adv_loss = self.adv_loss(fake, real)
            total_loss = total_loss + (self.adv_lambda * adv_loss) 

        return total_loss, lpips, tc, pred, image_0

    def train_epoch(self, epoch):

        self.model.train()
        self.discr.eval()
        lpips_sma = 0 # sma = simple muving average

        for i, sequence in enumerate(tqdm(self.train_dataloader, desc='training')):

            bs = sequence['img'].shape[0]
            for _ in range(self.mult):
                
                shuffled_seq = random.sample(range(bs), bs) # shuffle the mini batch 
                sequence['events'] = sequence['events'][shuffled_seq, ...].to(self.device)
                sequence['img'] = sequence['img'][shuffled_seq, ...].to(self.device)
                sequence['flow'] = sequence['flow'][shuffled_seq, ...].to(self.device)
                
                self.optm.zero_grad()
                total_loss, lpips_loss, tc_loss, output, img = self.train_sequ(sequence, epoch)
                self.accelerator.backward(total_loss)
                self.optm.step()      
                  
            with torch.no_grad():
                global_step = (self.len_epoch * epoch) + i
                self.writer.add_scalar('Glb_step', global_step, global_step)
                self.writer.add_scalar('lr', self.optm.state_dict()['param_groups'][0]['lr'], global_step)
                self.writer.add_scalar('train/lpips', lpips_loss.item(), global_step)
                self.writer.add_scalar('train/tc', tc_loss.item(), global_step)
                self.writer.add_scalar('train/loss', total_loss.item(), global_step)
                
                # moving average = [x_(n+1) + (n * CA_(n))] / (n + 1)
                lpips_sma = (lpips_loss.item() + (i * lpips_sma)) / (i + 1)

        self.writer.add_image('train/pred', torch.clip(output[0], 0, 1), global_step)
        self.writer.add_image('train/GT', img[0], global_step)
        self.writer.add_scalar('train/lpips_sma', lpips_sma, global_step)
        if epoch < self.sche_max:
            self.scheduler.step()   

        
    @torch.no_grad()
    def valid_epoch(self, epoch):
        
        tc_, ssim_, mse_, lpips_, total_loss_= [], [], [], [], []
        self.model.eval()
        idx0 = -1
        idx1 = -1
        
        for sequence in tqdm(self.val_dataloader, desc='validation'):

            idx0 = idx1
            idx1 = sequence['name'][0]
            # reset params in a new sequence
            if idx0 != idx1:
                tc, ssim, mse, lpips, total_loss = [], [], [], [], []
                seq_len = sequence['len'].item()
                sensor_size = [*sequence['frame'].shape[-2:]]
                self.metrics.set_params(sensor_size)
                padding = padd_fn(sensor_size)
            
                self.model.zero_grad()
                self.metrics.flownet.zero_grad()
                self.metrics.lpips.zero_grad()
                self.reset_state()
            
            image = sequence['frame'].to(self.device)
            v_len = sequence['events'].shape[1]

            for i in range(v_len):
                voxel = padding.padding(sequence['events'][:, i])
                voxel = voxel.to(self.device)
                pred = torch.clip(self.model(voxel), 0, 1)
                pred = padding.unpadd(pred)

            if idx0 != idx1:
                self.metrics.target0 = image
                self.metrics.pred0 = pred
                continue
            
            tc_metric = self.metrics.tc_metric(pred, image)
            ssim_metric = self.metrics.ssim_metric(pred, image)
            mse_metric = self.metrics.mse_metric(pred, image)
            lpips_metric = self.metrics.lpips_metric(pred, image)

            tc.append(tc_metric)
            ssim.append(ssim_metric)
            mse.append(mse_metric)
            lpips.append(lpips_metric)
            
            if len(tc) >= seq_len - 1:
                seq_name = sequence['name'][0]

                global_step = (self.len_epoch * epoch) + len(self.train_dataloader)
                self.writer.add_image(f'val/{seq_name}/pred', pred[0], global_step)
                self.writer.add_image(f'val/{seq_name}/GT', image[0], global_step)
                
                tc = torch.mean(torch.tensor(tc))
                mse = torch.mean(torch.tensor(mse))
                ssim = torch.mean(torch.tensor(ssim))
                lpips = torch.mean(torch.tensor(lpips))
                total_loss = lpips + mse + (2 * tc)

                self.writer.add_scalar(f'val/{seq_name}/tc', tc.item(), global_step)
                self.writer.add_scalar(f'val/{seq_name}/mse', mse.item(), global_step)
                self.writer.add_scalar(f'val/{seq_name}/ssim', ssim.item(), global_step)
                self.writer.add_scalar(f'val/{seq_name}/lpips_sma', lpips.item(), global_step)
                self.writer.add_scalar(f'val/{seq_name}/loss', total_loss.item(), global_step)

                tc_.append(tc.item())
                mse_.append(mse.item())
                ssim_.append(ssim.item())
                lpips_.append(lpips.item())
                total_loss_.append(total_loss.item())

        global_step = (self.len_epoch * epoch) + len(self.train_dataloader)
        self.writer.add_scalar('val/z/tc', torch.mean(torch.tensor(tc_)), global_step)
        self.writer.add_scalar('val/z/mse', torch.mean(torch.tensor(mse_)), global_step)
        self.writer.add_scalar('val/z/ssim', torch.mean(torch.tensor(ssim_)), global_step)
        self.writer.add_scalar('val/z/lpips_sma', torch.mean(torch.tensor(lpips_)), global_step)
        self.writer.add_scalar('val/z/loss', torch.mean(torch.tensor(total_loss_)), global_step)

        return torch.mean(torch.tensor(lpips_))

    def save_ckpt(self, epoch, lpips_sma):
        
        if self.debug:
            arch = type(self.model).__name__
            state_dict = self.model.state_dict()
        else:
            arch = type(self.model.module).__name__
            state_dict = self.model.module.state_dict()
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optm.state_dict(),
            'lpips': lpips_sma,
            'best_lpips': self.best_lpips,
        }

        if lpips_sma < self.best_lpips:
            self.best_lpips = lpips_sma
            best_path = self.ckpt_path + f'/best_model_@_{epoch:09}.pth'
            state['best_lpips'] = lpips_sma
            torch.save(state, best_path) 

        filename = str(self.ckpt_path + f'/checkpoint-epoch_{epoch:09}.pth')
        torch.save(state, filename)
    

    @torch.no_grad()
    def infer_sequ(self, sequence):

        seq_len = sequence['events'].shape[1]
        self.reset_state()
        # the last image is not used in the loop
        for i in range(seq_len - 1):
            
            voxel = sequence['events'][:, i]
            pred = self.model(voxel)
        
        return sequence['img'][:, -2], pred

    def train_adv(self, epoch):

        self.model.eval()
        self.discr.train()
        for sequence in tqdm(self.train_dataloader, desc='train discr'):
            
            img, pred = self.infer_sequ(sequence)
            self.optm_d.zero_grad()

            fake_pred = self.discr(pred.detach())
            real_pred = self.discr(img)

            fake_loss = self.adv_loss(fake_pred, 0. * torch.ones_like(fake_pred))
            real_loss = self.adv_loss(real_pred, 1. * torch.ones_like(real_pred))

            discr_loss = (fake_loss + real_loss) / 2

            self.accelerator.backward(discr_loss)
            self.optm_d.step()   
        
        
        state = {
            'epoch': epoch,
            'state_dict': self.discr.state_dict(),
            'optimizer': self.optm_d.state_dict(),
        }

        filename = str(self.ckpt_path + f'/disc-epoch_{epoch:09}.pth')
        torch.save(state, filename)


    def train(self, ):

        for epoch in trange(self.start_epoch, self.epochs, 1, desc='epochs'):

            self.train_epoch(epoch)
            epch = 1 + epoch

            # train discriminator at aech freq 
            if (epch % self.adv_freq) == 0 and epoch > (self.init_discr - 2):
                self.train_adv(epch)

            # validate at each validation time and at the last epoch
            if (epch % self.valid_freq) == 0 or (epch >= self.epochs): 
                lpips = self.valid_epoch(epoch)
                self.save_ckpt(epoch, lpips)
            

def main(args):

    seed_everything(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('E2VID m2m training')
    parser.add_argument('--train_data_path', default='./tr_m2o_data1')
    parser.add_argument('--ckpt_path', default='./test_m2o_train', type=str)
    
    # device parames
    parser.add_argument('--num_workers', default=2, type=int) 
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--dl_mult', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', default=False, type=bool)
    

    # training hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--discr_lr', default=1e-4, type=float)
    parser.add_argument('--tc_lambda', default=2, type=float)
    parser.add_argument('--ss_lambda', default=1, type=float)
    parser.add_argument('--adv_lambda', default=1, type=float)
    parser.add_argument('--rec_lambda', default=0.2, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--sche_max', default=1000, type=int)
    parser.add_argument('--init_discr', default=1000, type=int)
    parser.add_argument('--valid_freq', default=2, type=int)
    parser.add_argument('--adv_freq', default=1, type=int)
    parser.add_argument('--transforms', default={'randomCrop':{'size':112}, 'randomFlip':{'h_flip':0.5, 'v_flip':0.5}})
    parser.add_argument('--ev_rate', default=0.1)
    
    
    # net hyperparameters
    parser.add_argument('--num_bins', default=5, type=int)
    parser.add_argument('--base_num_channels', default=32, type=int)
    parser.add_argument('--kernel', default=5, type=int)
    parser.add_argument('--num_residual_blocks', default=2, type=int)
    parser.add_argument('--num_encoders', default=3, type=int)
    parser.add_argument('--num_output_channels', default=1, type=int)

    # test params
    parser.add_argument('--val_data_path', default='./val_data', type=str)
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()
    main(args)