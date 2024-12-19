import numpy as np
import torch
from tqdm import tqdm
import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
        def __init__(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patientce=patience
        self.counter=0
        self.mode = mode
        self.best_score=None
        self.early_stop=False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
    def __call__(self, epoch_score, model, model_path):
        if self.mode=='min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score<self.best_score+self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class Engine:
    @staticmethod
    def train(fold, epoch, data_loader, lossF, model, optimizer, device):
        losses = AverageMeter()
        accuracies = AverageMeter()
        model.train()
        bar = tqdm(data_loader)
        for batch, (data, target) in enumerate(bar):
            iter = len(bar)
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            out = model(data)
            # out = model(data, target.long())
            loss = lossF(out, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
            accuracy = (predictions == target.detach().cpu().numpy()).mean()
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy.item(), batch_size)
            bar.set_description(
                "Fold%d, Epoch%d : Ave train loss: %.6f, the accu trainset is %.4f " % (
                fold, epoch, losses.avg, accuracies.avg))
        return accuracies.avg, losses.avg

    @staticmethod
    def evaluate(fold, epoch, data_loader, lossF, model, device):
        accuracies = AverageMeter()
        losses = AverageMeter()
        top3_accuracies = AverageMeter()
        model.eval()   # 不启用batch normalization 和 dropout
        bar = tqdm(data_loader)
        with torch.no_grad():
            for i, (data, target) in enumerate(bar):
                batch_size = data.size(0)
                data, target = data.to(device), target.to(device)
                out = model(data)
                _, label_len = out.shape
                loss = lossF(out, target.long())
                predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
                target_num = target.detach().cpu().numpy()
                # 计算top3准确率
                _, predictions_top3 = torch.topk(out, 3, dim=1)
                top3_num = predictions_top3.cpu().numpy().squeeze()
                mth, nth = top3_num.shape
                top3_prediction = np.zeros(mth)
                for ith in range(nth):
                    top3_temp = (top3_num[:, ith].flatten() == target_num)
                    top3_prediction += top3_temp
                accuracy_top3 = top3_prediction.mean()
                accuracy = (predictions == target_num).mean()
                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy.item(), batch_size)
                top3_accuracies.update(accuracy_top3.item(), batch_size)
                bar.set_description(
                    "Fold%d, Epoch%d,: Ave val loss is %.6f, the accu val is %.4f, the top3 accu val is %.4f"
                    % (fold, epoch, losses.avg, accuracies.avg, top3_accuracies.avg))
        return accuracies.avg, losses.avg, top3_accuracies.avg

    def test(self, lossF, model, device):
        accuracies = AverageMeter()
        losses = AverageMeter()
        model.eval()
        bar = tqdm(self)
        probs = []
        predicticted = []
        true = []
        with torch.no_grad():
            for i, (data, target) in enumerate(bar):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                out = model(data)
                # loss = lossF(out, target.long())
                predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
                prob = torch.softmax(out, dim=-1).detach().cpu().numpy()
                # accuracy = (predictions == target).mean()
                # losses.update(loss.item(), batch_size)
                # accuracies.update(accuracy.item(), batch_size)
                bar.set_description(
                    "Average test loss is %.6f, the accuracy rate of testset is %.4f " % (losses.avg, accuracies.avg))
                predicticted += predictions.tolist()
                true += target.tolist()
                probs += prob.tolist()
        return np.array(probs), np.array(predicticted), np.array(true)
