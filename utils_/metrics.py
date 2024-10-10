import torch
import torchvision
import torch.nn as nn
from utils_.PerceptualSimilarity import models
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


        
class metrics_Fn:
    def __init__(self, net, device):
        
        self.device = device
        self.MSE = nn.MSELoss()
        self.SSIM = ssim
        self.data_range = 1
        # self.model = VGG19().to(device).eval()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.model = models.PerceptualLoss(net=net, use_gpu=device)

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.flownet = raft_small(weights=weights, progress=False).to(self.device).eval()
        self.alpha = 50

    @torch.no_grad()
    def set_params(self, sensor_size):
        
        self.height, self.width = sensor_size
        # should be divisible by 8 the img size
        # padding is added in (8/2=4) two sized
        mod = (self.width % 8) / 8
        padd_w = 0 if mod == 0 else 1 - mod 
        self.w1 = int(padd_w * 8)

        mod = (self.height % 8) / 8
        padd_h = 0 if mod == 0 else 1 - mod 
        self.h1 = int(padd_h * 8)

        xx, yy = torch.meshgrid(torch.arange(self.width), torch.arange(self.height))
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        self.xx, self.yy = xx.to(self.device), yy.to(self.device)
        self.pred0 = None
        self.target0 = None
        

    @torch.no_grad()
    def tc_metric(self, pred1, target1):
        self.val_assert(pred1, target1)
        
        img0_, img1_ = self.transforms(torch.tile(self.target0, [1, 3, 1, 1]), torch.tile(target1, [1, 3, 1, 1]))
        img0_ = F.pad(img0_, (self.w1, 0, self.h1, 0)).to(self.device).to(pred1.device)
        img1_ = F.pad(img1_, (self.w1, 0, self.h1, 0)).to(self.device).to(pred1.device)
        flow = self.flownet(img0_, img1_)[-1]  # forward optical flow
        if self.h1 > 0:
            flow = flow[:, :, self.h1:, :]
        if self.w1 > 0:
            flow = flow[:, :, :, self.w1:]
        flow_x = flow[:, 0, :, :].to(pred1.device)  # N x H x W
        flow_y = flow[:, 1, :, :].to(pred1.device)  # N x H x W

        warping_grid_x = self.xx - flow_x  # N x H x W
        warping_grid_y = self.yy - flow_y  # N x H x W

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
    def lpips_metric(self, pred, target, normalize=True):
        self.val_assert(pred, target)
        lpips = self.model.forward(pred, target, normalize=normalize).mean()
        return lpips.item()
    
    @torch.no_grad()
    def lpips_metric2(self, pred, target):

        self.val_assert(pred, target)

        target = (target * 2) - 1
        pred = (pred * 2) - 1
        x_vgg, y_vgg = self.vgg(pred), self.vgg(target.detach())
        f_loss = 0
        for i in range(len(x_vgg)):
            f_loss += self.weights[i] * self.MSE(x_vgg[i], y_vgg[i].detach())
        
        return f_loss.item() / len(x_vgg)
    
    def ssim_metric(self, pred, target):
        self.val_assert(pred, target)
        pred = pred[0, 0].detach().cpu().numpy()
        target = target[0, 0].detach().cpu().numpy()
        return 1 - self.SSIM(pred, target, data_range=self.data_range)

    @torch.no_grad()
    def mse_metric(self, pred, target):
        self.val_assert(pred, target)
        return self.MSE(pred, target).item()
    
    @staticmethod
    def val_assert(pred, target):
        assert pred.max() <= 1
        assert target.max() <= 1

        assert pred.min() >= 0
        assert target.min() >= 0