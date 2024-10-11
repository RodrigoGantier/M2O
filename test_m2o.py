import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import numpy as np
import random
import numbers
import os
from tqdm import trange
from tqdm import tqdm
import cv2
from math import floor, ceil

from utils_.test_dataset import testDataLoad
from torch.utils.data import DataLoader
from models_.e2vid import UConvLSTM
from utils_.metrics import metrics_Fn


def seed_everything(seed):
    import torch
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')


class padd_fn:
    def __init__(self, net_scale):

        self.net_scale = net_scale
        self.w = None
        self.h = None

    def set_params(self, img_size):
        self.w = img_size[1]
        self.h = img_size[0]

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
    

class Testet:
    def __init__(self, args):
        
        self.model = UConvLSTM(args)
        self.padd = padd_fn(8)

        test_dataset = testDataLoad(args)
        self.test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

        self.writer = SummaryWriter(args.tensorbloard_path)
        self.ckpt_path = args.ckpt_path
        self.tensorbloard_path = args.tensorbloard_path

        self.metrics = metrics_Fn(net='vgg', device=args.device)

        self.device = args.device
        self.verbose = args.verbose
        assert os.path.isfile(self.ckpt_path), 'No Check point found'
        self.resume_checkpoint()
        self.model.to(self.device)

        self.vid_path = 'vid_output'
        if not os.path.exists(self.vid_path):
            os.makedirs(self.vid_path)

    def resume_checkpoint(self, ):

        resume_path = self.ckpt_path

        print(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')

        self.model.load_state_dict(checkpoint, strict=True)
        print(f"Model loaded")   

    def reset_state(self, ):
        for j in range(self.model.num_encoders):
            self.model.encoders[j].recurrent_block.state = None

    @torch.no_grad()
    def final_test(self, ):
        
        if self.verbose:
            name_win = 'evs'
            cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)
        
        tc_, ssim_, mse_, lpips_, total_loss_ = [], [], [], [], []

        self.model.eval()
        idx0 = -1
        idx1 = -1
        img_num = 0
        for sequence in tqdm(self.test_dataset, desc='Final Test'):

            idx0 = idx1
            idx1 = sequence['name'][0]
            # restar parameters in a new sequence
            if idx0 != idx1:
                
                tc, ssim, mse, lpips = [], [], [], []
                seq_len = sequence['len'].item()
                sensor_size = [*sequence['frame'].shape[-2:]]
                self.metrics.set_params(sensor_size)
                self.padd.set_params(sensor_size)
                self.reset_state()

                vid_name = os.path.join(self.vid_path, idx1 + '.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out_vid = cv2.VideoWriter(vid_name, fourcc, 30.0, (sensor_size[1], sensor_size[0]))
            
            image = sequence['frame'].float().to(args.device)
            v_len = sequence['events'].shape[1]

            for i in range(v_len): 
                    
                voxel = self.padd.padding(sequence['events'][:, i])
                voxel = voxel.float().to(args.device)
                pred = self.model(voxel)[:, 0:1] 
                pred = self.padd.unpadd(pred)
                pred = torch.clip(pred, 0, 1.0)

            if idx0 != idx1:
                self.metrics.target0 = image
                self.metrics.pred0 = pred
                continue
                
            tc_metric = self.metrics.tc_metric(pred, image)
            ssim_metric = self.metrics.ssim_metric(pred, image)
            mse_metric = self.metrics.mse_metric(pred, image)
            lipsp_metric = self.metrics.lpips_metric(pred, image)
            is_nan = np.isnan(tc_metric) + np.isnan(ssim_metric) + np.isnan(mse_metric) + np.isnan(lipsp_metric)
            assert np.isnan(is_nan) == False

            frame = np.tile(pred[0, 0, :, :, None].detach().cpu().numpy(), (1, 1, 3)) * 255
            frame = np.asarray(frame, dtype=np.uint8)
            out_vid.write(frame)
                
            tc.append(tc_metric)
            ssim.append(ssim_metric)
            mse.append(mse_metric)
            lpips.append(lipsp_metric)

            if self.verbose:
                output = pred[0].permute(1, 2, 0).detach().cpu().numpy()
                output = np.tile(output, [1, 1, 3])
                
                img = image[0].permute(1, 2, 0).detach().cpu().numpy()
                img = np.tile(img, [1, 1, 3])
                img_out = np.concatenate([img, output], 1)
                img_out = cv2.resize(img_out, (1440, 540))
                cv2.imshow(name_win, img_out)
                cv2.waitKey(1)

                img_num += 1
                if img_num == 723:
                    print('stop')
                    
            if len(tc) >= seq_len - 1:
                sequ_name = sequence['name'][0]
                self.writer.add_image(f'{sequ_name}/pred', pred[0], 0)
                self.writer.add_image(f'{sequ_name}/GT', image[0], 0)

                tc = np.mean(tc)
                mse = np.mean(mse)
                ssim = np.mean(ssim)
                lpips = np.mean(lpips)
                total_loss = lpips + mse + (2 * tc)

                self.writer.add_scalar(f'{sequ_name}/tc', tc, 0)
                self.writer.add_scalar(f'{sequ_name}/mse', mse, 0)
                self.writer.add_scalar(f'{sequ_name}/ssim', ssim, 0)
                self.writer.add_scalar(f'{sequ_name}/lpips_sma', lpips, 0)
                self.writer.add_scalar(f'{sequ_name}/loss', total_loss, 0)

                tc_.append(tc)
                mse_.append(mse)
                ssim_.append(ssim)
                lpips_.append(lpips)
                total_loss_.append(total_loss)

        self.writer.add_scalar(f'Mean/tc', np.mean(tc_), 0)
        self.writer.add_scalar(f'Mean/mse', np.mean(mse_), 0)
        self.writer.add_scalar(f'Mean/ssim', np.mean(ssim_), 0)
        self.writer.add_scalar(f'Mean/lpips_sma', np.mean(lpips_), 0)
        self.writer.add_scalar(f'Mean/loss', np.mean(total_loss_), 0)
        
        print('Finished')


def main(args):

    seed_everything(args.seed)
    tester = Testet(args)
    tester.final_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gradient prediction')
    parser.add_argument('--test_data_path', default='./datasets_path/test_night')
    # parser.add_argument('--test_data_path', default='./datasets_path/MVSEC')
    # parser.add_argument('--test_data_path', default='./datasets_path/HQF')
    # parser.add_argument('--test_data_path', default='./datasets_path/ECD')
    # parser.add_argument('--test_data_path', default='./datasets_path/test')
    parser.add_argument('--test_data_json', default='./datasets_path/dict.json')

    
    parser.add_argument('--ckpt_path', default='./ckpt_m2o_adv_test/checkpoint-m2o.pth')
    parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2_lol')
    # parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2_MVSEC')
    # parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2best_HQF')
    # parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2_EDC')
    # parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2_night')
    # parser.add_argument('--tensorbloard_path', default='./tensorbloard/m2o_noise_adv2')

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--cuda_ids', default=[0, ], type=str)
    parser.add_argument('--verbose', default=False, type=bool)
    
    # net parameters
    parser.add_argument('--in_chans', default=5)
    parser.add_argument('--embed_dim', default=16)
    parser.add_argument('--out_chans', default=1)
    
    parser.add_argument('--channels_base', default=16)
    parser.add_argument('--kernel', default=5)
    parser.add_argument('--num_bins', default=5)
    parser.add_argument('--skip_type', default='sum')
    parser.add_argument('--recurrent_block_type', default='convlstm')
    parser.add_argument('--num_encoders', default=3)
    parser.add_argument('--base_num_channels', default=32)
    parser.add_argument('--num_residual_blocks', default=2)
    parser.add_argument('--use_upsample_conv', default=True)
    parser.add_argument('--norm', default='none')
    parser.add_argument('--num_output_channels', default=1)
    
    parser.add_argument('--dim_encod', default=64)
    parser.add_argument('--diff', default=32, type=int)
    parser.add_argument('--dropout', default=1e-1, type=float)
    
    # train parametes
    parser.add_argument('--ev_rate', default=0.1)

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    main(args)
