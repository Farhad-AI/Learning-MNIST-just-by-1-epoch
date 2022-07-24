from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules import conv
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader


class CNN_MNIST(nn.Module):
    def __init__(self, ):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(1, 5, 3, 1),
                                  nn.ReLU(),
                                  nn.Dropout2d(p=0.1),
                                  nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(5) 
                                        )
        
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(5, 15, 3, 1),
                                  nn.ReLU(),
                                  nn.Dropout2d(p=0.1),
                                  nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(15) 
                                        )

        self.fc = nn.Linear(15*5*5, 300)
        self.output_layer = nn.Linear(300, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.reshape(x, (-1, 15*5*5))
        x = F.relu(self.fc(x))
        out = F.relu(self.output_layer(x))
        return out


model = CNN_MNIST()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="MNIST/", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="MNIST/", train=False, download=True, transform=transform)

train_losses, test_losses = [], []
train_correct, test_correct = [], []

epochs = 10
for i in range(1, epochs+1):
    trn_crr, tst_crr = 0, 0
    train_loader = DataLoader(train_data, batch_size=15, shuffle=True)
    
    model.train()
    for b, (x_trn, y_trn) in enumerate(train_loader):
        y_pred = model(x_trn)
        loss = criterion(y_pred, y_trn)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_crr = (predicted == y_trn).sum()
        trn_crr += batch_crr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_correct.append(trn_crr)
    print("epoch :", i, "==> train_acc :", 100*trn_crr/60000)

    # test
    test_loader = DataLoader(test_data, shuffle=True)
    with torch.no_grad():
        model.eval()
        for b, (x_tst, y_tst) in enumerate(test_loader):            
            y_eval = model(x_tst)
            predicted = torch.max(y_eval.data, 1)[1]
            batch_crr = (predicted == y_tst).sum()
            tst_crr += batch_crr

            loss = criterion(y_pred, y_trn)
            test_losses.append(loss)
            test_correct.append(tst_crr)
        print("epoch :", i, "==> test_acc :", 100*tst_crr/10000)
        print("-----------------------------------------")