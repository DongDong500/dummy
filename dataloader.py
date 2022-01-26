import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class dummydata(Dataset):

    def __init__(self, c=3, w=1920, h=1080) -> None:
        super(dummydata, self).__init__()
        self.channel = c
        self.width = w
        self.height = h

        self.labels = np.random.randint(4, size=160000)

    def __getitem__(self, index):
        
        buffer = torch.rand(self.channel*self.width*self.height)
        label = torch.randint(low=0, high=4, size=(1,))
        label = label.squeeze()

        return buffer, label

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":

    train_data = dummydata()
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels.size())
        
        if i == 1:
            break

