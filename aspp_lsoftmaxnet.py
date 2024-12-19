import torch.nn as nn
import torch.nn.functional as F
import torch
from lsoftmax import LSoftmaxLinear

# capture and combine multi-scale Raman features
class MultiScaleBlock(nn.Module):
    def __init__(self, inc, ouc, branchs=15, stride=1, reduction=16):
        super(MultiScaleBlock, self).__init__()
        self.inconvs = nn.ModuleList([])
        for branch in range(branchs):
            self.inconvs.append(nn.Sequential(
                nn.Conv1d(inc, ouc, kernel_size=3 + branch * 2, padding=branch+1, stride=stride, bias=False),
                nn.BatchNorm1d(ouc, eps=0.001, momentum=0.01),
            ))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Linear(ouc*branchs, ouc*branchs//reduction),
            nn.PReLU(),
            nn.Linear(ouc*branchs//reduction, ouc),
            nn.Sigmoid(),
        )
        self.ouconvs = nn.Sequential(
            nn.Conv1d(branchs*ouc, ouc, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(ouc, eps=0.001, momentum=0.01),
        )
        self.prelu = nn.Sequential(
            nn.PReLU()
        )
    def forward(self, x):
        for i, conv in enumerate(self.inconvs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        z = self.pool(feas).squeeze(-1)
        mask = self.se(z).unsqueeze(-1)
        out = self.ouconvs(feas)
        out = mask*out
        out = self.prelu(out)
        return out

class ScaleAdaptiveNet(nn.Module):
    def __init__(self, margin, device, num_classes=30):
        super(ScaleAdaptiveNet, self).__init__()
        self.margin = margin
        self.device = device
        self.feature = nn.Sequential(
            MultiScaleBlock(1, 32, stride=2),     # 500
            MultiScaleBlock(32, 64, stride=2),    # 250
            MultiScaleBlock(64, 128, stride=2),    # 125
            MultiScaleBlock(128, 256, stride=2),   # 63
            MultiScaleBlock(256, 512, stride=2),  # 32
            nn.Conv1d(512, 128, 1, 1, bias=False),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.Dropout(0.5)
        )
        # self.classify = nn.Linear(1024, num_classes)
        self.classify = nn.Sequential(
            nn.Linear(128*32, 4096),
            nn.BatchNorm1d(4096, eps=0.001, momentum=0.1),
            nn.PReLU(),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, eps=0.001, momentum=0.1),
        )
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=4096, output_features=num_classes, margin=margin, device=self.device)
        self.reset_parameters()
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        out = self.feature(x)
        out = out.flatten(start_dim=1)
        out = self.classify(out)
        out = self.lsoftmax_linear(input=out, target=target)
        return out

    # get Grad-CAMs of given spectra and targets
    def getgradCAM(self, x, target):
        self.eval()
        f = self.feature(x).detach()
        f.requires_grad_()
        out = f.flatten(start_dim=1)
        out = self.classify(out)
        onehot = torch.zeros(out.shape).to(x.device)
        for i in range(len(target)):
            onehot[i][target[i].long().item()]=1
        out = onehot*out
        out = out.sum()
        out.backward()
        mask = f.grad
        mask = mask.mean(-1, keepdim=True)
        cam = (f*mask).sum(dim=1)
        cam = F.relu(cam).unsqueeze(1)
        cam = F.interpolate(cam, scale_factor=x.shape[-1]/cam.shape[-1], mode='linear',
                            align_corners=True, recompute_scale_factor=True)
        cam = (cam-cam.min())/(cam.max()-cam.min())
        return cam.squeeze(1)