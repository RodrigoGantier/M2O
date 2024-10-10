import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights
from collections import namedtuple
from skimage.metrics import structural_similarity
import numpy as np

        

class metricsFn:
    def __init__(self, lpips_model, flow_model, alpha=50):
        # lpips model
        self.lpips = lpips_model
        # flow model and corresponding transformations
        self.flownet = flow_model
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        # MSE and SSIM loss
        self.MSE = torch.nn.MSELoss()
        self.SSIM = structural_similarity
        self.data_range = 1 #for ssim (1-0)
        # alpha for the oclutino mask (tc_loss)
        self.alpha = alpha
        self.height = None
        self.width = None

    @torch.no_grad()
    def set_params(self, sensor_size):
        
        self.height, self.width = sensor_size
        # should be divisible by 8 the img size
        # if needed, padding is added on one sized
        mod = (self.width % 8) / 8
        padd_w = 0 if mod == 0 else 1 - mod 
        self.w1 = int(padd_w * 8)

        mod = (self.height % 8) / 8
        padd_h = 0 if mod == 0 else 1 - mod 
        self.h1 = int(padd_h * 8)

        self.xx, self.yy = torch.meshgrid(
            torch.arange(self.width), 
            torch.arange(self.height))
        
        self.xx.transpose_(0, 1)
        self.yy.transpose_(0, 1)
        self.pred0 = None
        self.target0 = None

    @torch.no_grad()
    def tc_metric(self, pred1, target1):
        # asset the imgs are in 1-0 range values
        self.value_assert(pred1, target1)
        # transform imgs
        img0_, img1_ = self.transforms(
            torch.tile(self.target0, [1, 3, 1, 1]), 
            torch.tile(target1, [1, 3, 1, 1]))
        # padding, img dimension should be divisible by 8
        img0_ = F.pad(img0_, (self.w1, 0, self.h1, 0)).to(pred1.device)
        img1_ = F.pad(img1_, (self.w1, 0, self.h1, 0)).to(pred1.device)

        flow = self.flownet(img0_, img1_)  # forward optical flow
        # reverse the padding
        if self.h1 > 0:
            flow = flow[:, :, self.h1:, :]
        if self.w1 > 0:
            flow = flow[:, :, :, self.w1:]
        # extract x and y terms
        flow_x = flow[:, 0, :, :].to(pred1.device)  # N x H x W
        flow_y = flow[:, 1, :, :].to(pred1.device)  # N x H x W
        # backward warping 
        warping_grid_x = self.xx.to(pred1.device) - flow_x  # N x H x W
        warping_grid_y = self.yy.to(pred1.device) - flow_y  # N x H x W
        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (self.width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (self.height - 1)) - 1
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2
        # warp last GT image to the penultimate: image (n) to (n-1)
        target1_warped_to0 = F.grid_sample(target1, warping_grid, align_corners=True)
        # warp  predited image(n) to the (n - 1) 
        prod1_warped_to0 = F.grid_sample(pred1, warping_grid, align_corners=True)
        # compure the visibility_mask with GT image(n-1) and GT image(n) warped to (n-1)
        visibility_mask = torch.exp(-self.alpha * (self.target0 - target1_warped_to0) ** 2)
        # calculate the distance between the last predicted image (n-1) and the present predited image(n) warped to (n-1)
        tc_loss = visibility_mask * torch.abs(self.pred0 - prod1_warped_to0) / (torch.abs(self.pred0) + torch.abs(prod1_warped_to0) + 1e-5)

        # update the last predicted and GT image
        self.pred0 = pred1.clone()
        self.target0 = target1.clone()

        return tc_loss.mean().item()
    
    @torch.no_grad()
    def lpips_metric(self, pred, target):
        self.value_assert(pred, target)
        lpips = self.lpips.forward(pred, target).mean()
        return lpips.item()
    
    def ssim_metric(self, pred, target):
        self.value_assert(pred, target)
        pred = pred[0, 0].detach().cpu().numpy()
        target = target[0, 0].detach().cpu().numpy()
        return 1 - self.SSIM(pred, target, data_range=self.data_range)
    
    @torch.no_grad()
    def mse_metric(self, pred, target):
        return self.MSE(pred, target).item()
    
    @staticmethod
    def value_assert(pred, target):
        assert pred.max() <= 1
        assert target.max() <= 1

        assert pred.min() >= 0
        assert target.min() >= 0
        

class flowFn(nn.Module):
    def __init__(self, ):
        super(flowFn, self).__init__()

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.flownet = raft_small(weights=weights, progress=False).eval()
    
    def forward(self, img0, img1):
        return self.flownet(img0, img1)[-1]  # forward optical flow
        
    
class tcLoss:
    "time constancy loss"
    def __init__(self, ):
        self.alpha = None
        self.t_width = None
        self.t_height = None
        self.last_t = None

    def set_params(self, sensor_size, alpha=50):
        self.alpha = alpha
        self.t_height, self.t_width = sensor_size
        
        self.xx, self.yy = torch.meshgrid(
            torch.arange(self.t_width), 
            torch.arange(self.t_height))  # xx, yy -> WxH
        
        self.xx.transpose_(0, 1)
        self.yy.transpose_(0, 1)

    def __call__(self, pred, img_0, img_1, flow):
                       
        # tenporal consistency loss
        flow_x = flow[:, 0, :, :]  # N x H x W
        flow_y = flow[:, 1, :, :]  # N x H x W

        warping_grid_x = self.xx.to(pred.device) + flow_x  # N x H x W
        warping_grid_y = self.yy.to(pred.device) + flow_y  # N x H x W

        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (self.t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (self.t_height - 1)) - 1
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        # warp the n GT image to the n+1: image (n) to (n+1)
        img0_warped_img1 = F.grid_sample(img_0, warping_grid, align_corners=True)
        # warp the n predited image to the n + 1: predic (n) to (n+1) 
        prod0_warped_to1 = F.grid_sample(pred, warping_grid, align_corners=True)

        # compure the visibility_mask with GT image(n+1) and the warped GT image(n+1)  
        visibility_mask = torch.exp(-self.alpha * (img_1 - img0_warped_img1) ** 2)

        # calculate the distance between the GT image(n+1) and the warped predited image(n+1)
        tc_loss = visibility_mask * torch.abs(img_1 - prod0_warped_to1) \
             / (torch.abs(img_1) + torch.abs(prod0_warped_to1) + 1e-5)

        return tc_loss.mean()
        
class m2m_loss():
    def __init__(self, sensor_size, device, net='alex', alpha=50):
        self.lpips_loss = lpips(model=net).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.alpha = alpha
        self.t_width, self.t_heght = sensor_size

        xx, yy = torch.meshgrid(torch.arange(self.t_width), torch.arange(self.t_heght))

        xx.transpose_(0, 1)
        yy.transpose_(0, 1)

        self.xx = xx
        self.yy = yy
        
        self.last_t = None
    
    def __call__(self,pred, trgt, flow):
        # MSE loss
        # mse = self.mse_loss(pred, trgt).to(pred.device)

        # temporal consistency loss
        flow_x = flow[:, 0, :, :] # N x H x W
        flow_y = flow[:, 1, :, :] # N x H x W
        
        warping_grid_x = self.xx.to(pred.device) - flow_x
        warping_grid_y = self.yy.to(pred.device) - flow_y

        # normalize warping grid to [-1, 1]
        warping_grid_x = (2 * warping_grid_x / (self.t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (self.t_heght - 1)) - 1
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        # warping last GT image to the penultimate image: image(n) -> image(n-1)
        image1_warped_to0 = F.grid_sample(trgt, warping_grid, align_corners=True)
        # warping predicted image to the penultimate image: pred(n) -> pred(n-1)
        pred1_warped_to0 = F.grid_sample(pred, warping_grid, align_corners=True)

        # compute the visibility mask with the GT image(n-1) and GT image(n) waped to image(n-1)
        visibiliti_mask = torch.exp(-self.alpha * (self.last_t - image1_warped_to0)**2)

        # calculate the distance between the GT image(n-1) and the predicted(n) warped to predicted(n - 1)
        tc_loss = visibiliti_mask * torch.abs(self.last_t - pred1_warped_to0) \
            / (torch.abs(self.last_t) + torch.abs(pred1_warped_to0) + 1e-5)
        
        lpips = self.lpips_loss(pred, trgt).to(pred.device)

        # return lpips, mse, tc_loss.mean().to(pred.device)
        return lpips, tc_loss.mean().to(pred.device)
        

class alex_(nn.Module):
    def __init__(self, ):
        super(alex_, self).__init__()

        net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), net[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), net[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), net[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), net[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), net[x])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out
    

class vgg_(nn.Module):
    def __init__(self, ):
        super(vgg_, self).__init__()

        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), net[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), net[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), net[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), net[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), net[x])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("AlexnetOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_3, h_relu5_3)

        return out


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.tensor([.485,.456,.406]).view(1, 3, 1, 1))
        self.register_buffer('scale', torch.tensor([.229,.224,.225]).view(1, 3, 1, 1))

    def forward(self, inp):
        return (inp - self.shift.to(inp.device)) / self.scale.to(inp.device)
        

class lpips(nn.Module):
    def __init__(self, model='alex'):
        super(lpips, self).__init__()
        self.net = alex_() if model == 'alex' else vgg_()
        self.scaling_layer = ScalingLayer()
        self.last_t = None
        self.L = self.net.N_slices

        self.net.eval()
    
    def forward(self, in0, in1):
        
        assert in0.shape[1] == in1.shape[1]

        if in0.shape[1] != 3:
            in0 = in0.repeat(1, 3, 1, 1)  
            in1 = in1.repeat(1, 3, 1, 1)

        in0 = self.scaling_layer(in0) 
        in1 = self.scaling_layer(in1) 

        outs0 = self.net.forward(in0)
        outs1 = self.net.forward(in1)

        val = 0
    
        for kk in range(self.L):
            diffs = (normalize_tensor(outs0[kk]) - normalize_tensor(outs1[kk]))**2
            val += spatial_average(torch.sum(diffs, dim=1, keepdim=True))
        
        return val.mean()


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.in_ch = in_ch
        
        conv_channels = [64, 128, 256]
        layers_dim = [self.in_ch] + conv_channels + [1]
        kernels = [4,4,4,4] 
        strides = [2,2,2,1]
        paddings = [1,1,1,1]
        activation = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out