import torch.optim.lr_scheduler as lr_scheduler
import os
from torch.utils.data import Dataset
class LossNotDecreasingLR:
    def __init__(self, optimizer, patience=30, factor=0.1, verbose=True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.last_update_epoch = None
        self.stop_training = False

    def step(self, epoch_loss, epoch):
        if self.best_loss is None:
            self.best_loss = epoch_loss
            self.last_update_epoch = epoch
            self.counter = epoch - self.last_update_epoch
        if epoch_loss > self.best_loss:  # loss没有下降
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\nLoss not decreasing for {self.patience} epochs. Reducing learning rate.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
                    print(param_group['lr'])
                    if param_group['lr'] <= 0.0000000001:
                        self.stop_training = True
                self.last_update_epoch = epoch
                #self.counter = epoch - self.last_update_epoch
                self.best_loss = epoch_loss
        else:
            self.best_loss = epoch_loss
            self.last_update_epoch = epoch

        self.counter = epoch - self.last_update_epoch

        return self.stop_training

def read_files_in_directory(root_list, size = 1):
    result_files = []

    for i in range(0, len(root_list)):
        index = 0
        for foldername, subfolders, filenames in os.walk(root_list[i]):
            for filename in filenames:
                if filename.endswith('.npz'):
                    file_path = os.path.join(foldername, filename)
                    if index % int(1/size) == 0:
                        result_files.append(file_path)
                    index = index+1
    return result_files


def read_files_in_directory_fx(root_list, size = 1, key = ''):
    result_files = []

    for i in range(0, len(root_list)):
        index = 0
        for foldername, subfolders, filenames in os.walk(root_list[i]):
            for filename in filenames:
                if filename.endswith(key):
                    file_path = os.path.join(foldername, filename)

                    if index % int(1/size) == 0:
                        result_files.append(file_path)
                    index = index+1
    return result_files


class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]

        return item1, item2, item3

class CustomDataset_2class(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]

        return item1, item2


class CustomDataset_4class(Dataset):
    def __init__(self, data1, data2, data3, data4):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]
        item4 = self.data4[index]

        return item1, item2, item3, item4


class CustomDataset_5class(Dataset):
    def __init__(self, data1, data2, data3, data4, data5):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]
        item4 = self.data4[index]
        item5 = self.data5[index]

        return item1, item2, item3, item4, item5