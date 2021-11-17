import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import csv
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--imsuffix', default='.jpg', type=str)
parser.add_argument('--mapping', default='/shared/rsaas/pka4/CV/Project/data/mappings.csv', type=str)
parser.add_argument('--qimgs', default='/shared/rsaas/pka4/CV/Project/data/query', type=str)
parser.add_argument('--refimgs', default='/shared/rsaas/pka4/CV/Project/data/ref', type=str)
parser.add_argument('--saveroot', default='output/', type=str)
parser.add_argument('--logfile', default='execution.log', type=str)
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
args, _ = parser.parse_known_args()

device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
print(device)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, transforms=None):
        self.pairs = pairs
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        qp, rp = self.pairs[i]
        Q = Image.open(f'{args.qimgs}/{qp}{args.imsuffix}').convert('RGB')
        if self.transforms is not None:
            Q = self.transforms(Q)
        if not rp == '':
            R = Image.open(f'{args.refimgs}/{rp}{args.imsuffix}').convert('RGB')
            if self.transforms is not None:
                R = self.transforms(R)
        else:
            R = 0 
        return (Q, R), torch.tensor(self.labels[i]).reshape(1,)
        

def read_mapping(fname):
    pairs = []
    with open(fname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        for row in csvreader: 
            pairs.append(row)
    return pairs

class Distance(nn.Module):
    def __init__(self, method): # Method is in ['learnt', 'ssd']
        super().__init__()
        self.layers = []
        self.method = method
        if method == 'learnt':
            self.layers.append(nn.Linear(8192*2, 2048))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(2048, 512))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(512, 128))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(128, 4))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(4, 1))
            self.layers.append(nn.Sigmoid())

            self.layers = nn.ModuleList(self.layers)

    def forward(self, x1, x2):
        if self.method == 'ssd':
            ssd = torch.linalg.norm(x1 - x2, axis=1).reshape(-1, 1)
            N = ssd.size()[0]
            for i in range(N):
                if ssd[i][0] < 1e-4:
                    ssd[i][0] = 0
                else:
                    ssd[i][0] = 1
            return ssd
            
        elif self.method == 'learnt':
            x = torch.cat([x1, x2], dim=1)
            for layer in self.layers:
                x = layer(x)
            return x
        
class MatchNet(nn.Module):
    def __init__(self, distance_method):
        super().__init__()
        self.encoder = torchvision.models.vgg16(pretrained=True).features # Keep it super simple for now
        # We may want to freeze a few layers and not retrain all from scratch
        self.distance = Distance(distance_method)

        # self.encoder = nn.ModuleList(self.encoder)
        # self.distance = nn.ModuleList(self.distance)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        e1 = e1.reshape(e1.shape[0], -1)
        e2 = e2.reshape(e2.shape[0], -1)

        return self.distance(e1, e2)


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(*x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

pairs = read_mapping(args.mapping)
y = []
# pairs = [p for p in pairs if not p[1] == ''] # Only taking the 4k+ matching pairs
for i in range(len(pairs)):
    y += [1.0] if pairs[i][1] == '' else [0.0]
    pairs[i] = (pairs[i][0], f'R{torch.randint(0, int(1e6), (1,))[0]:06d}') if pairs[i][1] == '' else pairs[i]
    

t = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128), # This needs to change
    transforms.ToTensor()
])

print(pairs[:5])

dataset = ImageDataset(pairs, y, transforms=t)
N = len(pairs)
trainset, testset, valset = random_split(dataset, [int(N*0.7), int(N*0.15), int(N*0.15)])

trainloader = DataLoader(trainset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)
testloader = DataLoader(testset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)
valloader = DataLoader(valset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)                                                

torch.manual_seed(42)
model = MatchNet('ssd').to(device)
loss_fn = nn.BCELoss() # TODO: Needs to be improved
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_step = make_train_step(model, loss_fn, optimizer)

# spath=newest(args.cppath)
# dccheckpoint = torch.load(spath[0])
# genmodel.load_state_dict(dccheckpoint['allstate_dict'])


n_epochs = 50
training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    trainloss = []
    for batch, y in tqdm(trainloader):
        Q, R = batch
        Q = Q.to(device)
        R = R.to(device)
        y = y.to(device)
        loss = train_step((Q, R), y)
        trainloss.append(loss)
    training_losses.append(np.array(trainloss).mean())

    with torch.no_grad():
        valloss = []
        for batch, y in tqdm(valloader):
            Q, R = batch
            Q = Q.to(device)
            R = R.to(device)
            y = y.to(device)
            model.eval()
            yhat = model(Q, R)
            loss = loss_fn(yhat, y).item()
            valloss.append(loss)
        validation_losses.append(np.array(valloss).mean())
    
    print(f"EPOCH {epoch}: Train loss: {training_losses[-1]:0.5f} \t Val loss: {validation_losses[-1]:0.5f}")

    if epoch % 5 == 0:
        directory = args.saveroot+'/'+'models/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        wname = directory + 'checkpoint-'+str(epoch)+'.pth.tar'
        state={'epoch': epoch,
            'allstate_dict': model.state_dict()}
        torch.save(state, wname)

fig, ax = plt.subplots()
ax.plot(range(len(training_losses)), training_losses, label='Train')
ax.plot(range(len(validation_losses)), validation_losses, label='Validation')
fig.savefig(f'{args.saveroot}/train_val_loss.png')
    
print("Testing")
with torch.no_grad():
    testloss = []
    for batch, y in tqdm(testloader):
        Q, R = batch
        Q = Q.to(device)
        R = R.to(device)
        y = y.to(device)
        model.eval()
        yhat = model(Q, R)
        loss = loss_fn(yhat, y).item()
        testloss.append(loss)
    print(f"Test loss: {np.array(testloss).mean():0.5f}")

print("Done")
    
