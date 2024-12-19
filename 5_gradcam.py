from sa_asppnet import ScaleAdaptiveNet
import matplotlib.pyplot as plt
from data import spectralloader
from tqdm import tqdm
import numpy as np
import torch
import time

device = 'cuda:0'
batch_size=100
def main():
    net = ScaleAdaptiveNet(1619)
    net.load_state_dict(torch.load('param/1619/pretrain/1.pth'))
    label = np.load('../data/Y_gram1000.npy')
    label = label[:, 0]
    label = label.tolist()
    LABEL_NAME = np.load('../data/gram_mineral_name.npy')
    LABEL_NAME = LABEL_NAME.tolist()
    t = spectralloader('gram1000', batch_size=batch_size, num_workers=1)
    data_all = np.zeros((5600, 1000))
    cam_all = np.zeros((5600, 1000))
    label_all = np.zeros(5600)
    for i, (data, target) in enumerate(tqdm(t)):
        data_all[i*100:(i+1)*100]=data.view(batch_size, 1000).detach().numpy()
        cam = net.getgradCAM(data, target)
        cam_all[i*100:(i+1)*100] = cam.view(batch_size, 1000).detach().numpy()
        label_all[i*100:(i+1)*100] = target.view(-1).detach().numpy()
    # data_all = np.concatenate([data_all[o:o+1] for o in label], axis=0)
    # cam_all = np.concatenate([cam_all[o:o+1] for o in label], axis=0)
    # label_all = np.concatenate([label_all[o:o+1] for o in label], axis=0)

    id = np.random.randint(0, 5600)
    print(id)
    data_single = data_all[id]
    cam_single = cam_all[id]
    cam_single = np.concatenate(([[cam_single] for i in range(60)]), axis=0)
    plt.subplots(2, 1, sharex=True, facecolor='white', figsize=(9, 4), dpi=500)
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(data_single, color='black')
    plt.title('Single Spectrum of '+LABEL_NAME[id])
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1000)
    ax2=plt.subplot(2, 1, 2)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(cam_single, interpolation='bilinear', cmap='jet')
    plt.show()


if __name__ == '__main__':
    main()