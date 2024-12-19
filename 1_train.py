import numpy as np
import torch
from sa_asppnet import ScaleAdaptiveNet
from data import spectralloader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn as nn
from utils import Engine, EarlyStop, seed_everything

device_ids = [0, 1]

CFG = {
    'dataset': 'mineral1000',  # choose dataset in reference,finetune,test,2018clinical and 2019clnical
    'device': 'cuda',
    'epoch': 100,   # maximun epochs for training
    'batch_size': 1024,
    'lr': 1e-4,
    'momentum': 0.9,
    'wd': 1e-5,
    'patience': 10,  # patience for early stop
    'sheduler_type': 'ReduceLROnPlateau',
    'seed': 42,
    'num_classes': 1343,
    'margin': 4,
    'accumulation_steps': 1,  # update parameters after forward and backward for 'accumulation_steps'times
    'num_folds': 5,
    'save_path': 'param/1343/pretrain/',  # saving the params of models to the path
    'acc_path': 'param/1343/acc_top1',
    'top3_acc_path': 'param/1343/acc_top3'
}

# Initialize data-folds
folds = spectralloader(
    dataset=CFG['dataset'],
    batch_size=CFG['batch_size'],
    num_folds=CFG['num_folds']
)


def main():
    # Config1
    for i in [1]:  # replaced with for i in folds:
        # trainloader = folds[i]['train']
        # val-loader = folds[i]['val']
        trainloader = folds['train']
        valloader = folds['val']
        net = ScaleAdaptiveNet(num_classes=CFG['num_classes']).to(CFG['device'])
        #    net.load_state_dict(torch.load('pretrained_model path')) use this if fine tune
        net = nn.DataParallel(net, device_ids=device_ids)
        loss_f = nn.CrossEntropyLoss().to(CFG['device'])
        optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
        es = EarlyStop(patience=CFG['patience'])
        val_acc_all = []
        top3_val_acc_all = []
        for epoch in range(1, CFG['epoch'] + 1):
            Engine.train(i, epoch, trainloader, loss_f, net, optimizer, CFG['device'])
            if epoch % 10 == 0:
                val_temp_acc, _, top3_val_temp_acc = \
                    Engine.evaluate(i, epoch, valloader, loss_f, net, CFG['device'])
                val_acc_all.append(val_temp_acc)
                top3_val_acc_all.append(top3_val_temp_acc)
        val_acc, _, top3_val_acc = Engine.evaluate(i, epoch, valloader, loss_f, net, CFG['device'])
        # save the model if val_acc is increasing.
        es(val_acc, net, CFG['save_path'] + str(i) + '.pth')
        np.savetxt(CFG['acc_path'] + '/' + 'val_acc_all.csv', val_acc_all)
        np.savetxt(CFG['top3_acc_path'] + '/' + 'top3_val_acc.csv', top3_val_acc_all)
        # if the val_acc stop increasing for 'patience' epoches, end the training for the fold
        if es.early_stop:
            break
    # After training, the model with highest validation accuracy on each fold will be saved in CFG
    #


if __name__ == '__main__':
    main()
