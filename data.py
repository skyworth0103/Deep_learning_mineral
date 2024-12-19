from transform import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from torchvision import transforms


class SpectralDataset(Dataset):
    def __init__(self, X, y, index_list=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        if index_list is None:
            self.index_list = np.arange(len(X))
        else:
            self.index_list = index_list

    def __getitem__(self, index):
        index = self.index_list[index]
        spectra, target = self.X[index], self.y[index]
        if self.transform:
            spectra = self.transform(spectra)
        return spectra, target

    def __len__(self):
        return len(self.index_list)


train_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomChoice([
        RandomBlur(),
        AddGaussianNoise(), ])], p=0.5),
    transforms.RandomApply([RandomDropout()], p=0.5),
    transforms.RandomApply([RandomScaleTransform()], p=0.5),
    ToFloatTensor()
])

valid_transform = ToFloatTensor()


def spectralloader(dataset, batch_size=64, num_workers=1, num_folds=5, seed=42):
    X = np.load('../data/X_' + dataset + '.npy')
    y = np.load('../data/Y_' + dataset + '.npy')
    y = y.flatten()
    x = np.arange(len(y))
    # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # x = np.arange(len(y))
    # fold = {}
    # i = 1
    # for train, val in skf.split(x, y):
    #     trainset = SpectralDataset(X, y, train, train_transform)
    #     valset = SpectralDataset(X, y, val, valid_transform)
    #     trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
    #                             pin_memory=True)
    #     valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    #                             pin_memory=True)
    #     fold[i] = {"train": trainloader, "val": valloader}
    #     i += 1
    # return fold
    train = x[:1000]
    val = x[1000:]
    test_ith = np.load('../data/test_ith.npy')
    train_ith = np.load('../data/train_ith.npy')
    # trainset = SpectralDataset(X, y, train, train_transform)
    # valset = SpectralDataset(X, y, val, valid_transform)
    trainset = SpectralDataset(X, y, train_ith, train_transform)
    valset = SpectralDataset(X, y, test_ith, valid_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                             pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                           pin_memory=True)
    fold = {"train": trainloader, "val": valloader}
    return fold
