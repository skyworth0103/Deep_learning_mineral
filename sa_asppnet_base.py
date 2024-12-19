import torch.nn as nn
import torch.nn.functional as F
import torch

# capture and combine multi-scale Raman features
class MultiBlock(nn.Module):
    def __init__(self, inc, depth, branchs=1, stride=1, reduction=16):
        super(MultiBlock, self).__init__()
        self.inconvs = nn.ModuleList([])
        self.mean = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels=inc, out_channels=depth, kernel_size=1, stride=1)  # k=1, s=1 no pad
        for branch in range(branchs):
            self.inconvs.append(nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=depth, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(depth, eps=0.001, momentum=0.01),
            ))
        self.se = nn.Sequential(
            nn.Linear(depth * branchs, depth * branchs // reduction),
            nn.PReLU(),
            nn.Linear(depth * branchs // reduction, depth),
            nn.Sigmoid(),
        )
        self.ouconvs = nn.Sequential(
            nn.Conv1d(depth*branchs, depth, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(depth, eps=0.001, momentum=0.01),
            nn.Conv1d(depth, depth, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(depth, eps=0.001, momentum=0.01),
        )
        self.prelu = nn.Sequential(
            nn.ReLU()
        )
    def forward(self, x):
        for i, conv in enumerate(self.inconvs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        z = self.mean(feas).squeeze(-1)
        mask = self.se(z).unsqueeze(-1)
        out = self.ouconvs(feas)
        out = mask * out
        out = self.prelu(out)
        return out

class ScaleNet(nn.Module):
    def __init__(self, num_classes=30):
        super(ScaleNet, self).__init__()
        self.feature = nn.Sequential(
            MultiBlock(1, 16, stride=2),     # 500
            MultiBlock(16, 32, stride=2),    # 250
            MultiBlock(32, 64, stride=2),    # 125
            MultiBlock(64, 128, stride=2),   # 63
            MultiBlock(128, 192, stride=2),  # 32
            nn.Conv1d(192, 64, 1, 1, bias=False),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.01),
        )
        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*32, 2048),
            nn.BatchNorm1d(2048, eps=0.001, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, eps=0.001, momentum=0.1),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )
    def forward(self, x):
        out = self.feature(x)
        out = out.flatten(start_dim=1)
        out = self.classify(out)
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