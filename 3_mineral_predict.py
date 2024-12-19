import torch
import numpy as np
from sa_asppnet import ScaleAdaptiveNet
from data import spectralloader
from utils import Engine
from data import valid_transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CFG = {
    'device': 'cuda',
    'batch_size': 128,
    'sheduler_type': 'ReduceLROnPlateau',
    'seed': 42,
    'num_classes': 1619,
    'save_path': 'param/1619/pretrain/',  # saving the params of models to the path
    'acc_path': 'param/1619/acc_top1',
    'top3_acc_path': 'param/1619/acc_top3',
    'real_mineral': 'param/1619/real_mineral_predict'
}

device = 'cuda:0'
def main():
    real_mineral = spectralloader('test')
    net = ScaleAdaptiveNet(num_classes=CFG['num_classes']).to(device)
    net.load_state_dict(torch.load('param/1619/pretrain/1.pth'))
    probs, pred, true = Engine.test(real_mineral, torch.nn.CrossEntropyLoss, net, device)
    #
    np.savetxt(CFG['real_mineral'] + '/' + 'real_mineral_predict_top1.csv', pred)
    np.savetxt(CFG['real_mineral'] + '/' + 'real_mineral_predict_true.csv', true)


if __name__ == '__main__':
    main()


