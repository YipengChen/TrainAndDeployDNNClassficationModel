import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# set parameter
epochs = 200
batch_size = 64
device = 'cpu'


# set path
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_dir_path, 'keypoint.csv')


# define Dataset
class MyDataset(Dataset):

    def __init__(self, ):
        self.x_data = np.loadtxt(data_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
        self.y_temp = np.loadtxt(data_path, delimiter=',', dtype='int32', usecols=(0))
        self.y_data = np.zeros((len(self.x_data), 7))
        for i in range(len(self.y_temp)):
            self.y_data[i, self.y_temp[i]] = 1
        self.x_data = torch.tensor(self.x_data, dtype=torch.float)
        self.y_data = torch.tensor(self.y_data, dtype=torch.float)
        print('x_data shape:', self.x_data.shape)
        print('y_data shape:', self.y_data.shape)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx].squeeze(), self.y_data[idx]


# define dataloder
train_dataloader = DataLoader(MyDataset(), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(MyDataset(), batch_size=batch_size)


# define model
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(21*2, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 7),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = MyModule().to(device)


# define loss function
loss_fn = nn.CrossEntropyLoss()


# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# start train
def train():
    for epoch in range(epochs):
        data_size = len(train_dataloader.dataset)

        model.train()
        for batch_index, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print("epoch:{}[{}/{}], loss:{}".format(
                    epoch, 
                    batch_index*batch_size, 
                    data_size, 
                    round(loss.item(),4)
                    )
                )

        model.eval()
        correct_num = 0
        for batch_index, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            result = np.equal(np.argmax(pred.detach().numpy(), axis=1),np.argmax(y.detach().numpy(), axis=1))
            correct_num += result.sum()

        print('epoch:{}, correct_num:{}'.format(epoch, correct_num))

train()