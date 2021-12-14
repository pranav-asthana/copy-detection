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

'''
 TODO
 1. Experiment with different number of layers frozen
 2. Experiment with different base nets
 3. Experiment with different LR, batch size, etc
'''


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
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--l2w', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--basenet', default='vgg', type=str)
parser.add_argument('--layers-frozen', default=10, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--load-model', default=False, type=bool)
parser.add_argument('--matches', default=4991, type=int)
args, _ = parser.parse_known_args()

device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
print(device)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, transforms=None):
        self.pairs = pairs
        self.labels = labels
        if labels == None:
            self.labels = np.zeros((len(pairs), ))
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

def newest(path):
    files=os.listdir(path)
    paths=[os.path.join(path, basename) for basename in files]
    return sorted(paths, key=os.path.getctime, reverse=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Distance(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.layers = []
        if indim >= 2048:
            self.layers.append(nn.Linear(indim*2, 2048))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(2048))
            self.layers.append(nn.Linear(2048, 512))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(512))
        else:
            self.layers.append(nn.Linear(indim*2, 512))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(512))
        
        self.layers.append(nn.Linear(512, 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(128))
        self.layers.append(nn.Linear(128, 4))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(4))
        self.layers.append(nn.Linear(4, 1))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
        
class MatchNet(nn.Module):
    def __init__(self, distance_method):
        super().__init__()

        self.distance_method = distance_method

        # We may want to freeze a few layers and not retrain all from scratch
        if args.basenet == 'vgg':
            # VGG-16
            self.encoder = torchvision.models.vgg16(pretrained=True).features
            for layer_num in range(args.layers_frozen): # Freeze first 10 layers
                for param in self.encoder[layer_num].parameters():
                    param.requires_grad = False
            indim = 8192

        if args.basenet == 'resnet18':
            # # ResNet-18
            self.encoder = torchvision.models.resnet18(pretrained=True)
            layer_num = 0
            for child in self.encoder.children():
                if layer_num == args.layers_frozen:
                    break
                for param in child.parameters():
                    param.requires_grad = False
                layer_num += 1
            self.encoder.fc = Identity()
            indim = 512
        
        if args.basenet == 'resnet50':
            # # ResNet-50
            self.encoder = torchvision.models.resnet50(pretrained=True)
            layer_num = 0
            for child in self.encoder.children():
                if layer_num == args.layers_frozen:
                    break
                for param in child.parameters():
                    param.requires_grad = False
                layer_num += 1
            self.encoder.fc = Identity()
            indim = 2048

        self.distance = Distance(indim)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        e1 = e1.reshape(e1.shape[0], -1)
        e2 = e2.reshape(e2.shape[0], -1)

        if self.distance_method == 'learnt':
            return self.distance(e1, e2)
        elif self.distance_method == 'ssd':
            ssd = torch.linalg.norm(e1 - e2, axis=1).reshape(-1, 1)
            N = ssd.size()[0]
            for i in range(N): # Batch
                if ssd[i][0] < 0.1: # Some arbitrary threshold
                    ssd[i][0] = 0
                else:
                    ssd[i][0] = 1
            return ssd

def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()

        yhat = model(*x) # For fully learnt classification model
        
        # # For descriptor + SSD model
        # e1 = model.encoder(x[0])
        # e2 = model.encoder(x[1])
        # yhat = torch.linalg.norm(e1 - e2, axis=1).reshape(-1, 1)
        # N = yhat.size()[0]
        # for i in range(N): # Batch
        #     if yhat[i][0] < 0.1: # Some arbitrary threshold
        #         yhat[i][0] = 0
        #     else:
        #         yhat[i][0] = 1
        # # EndIf

        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

pairs = read_mapping(args.mapping)
y = []

num_matches = 4991 # Known, but we can use a subset of known matches too
num_matches = args.matches
non_matches, matches = 0, 0
train_pairs = []
# for i in range(len(pairs)):
#     if pairs[i][1] == '':
#         if non_matches == num_matches: # Work with subset such that both classes are equally distributed
#             continue
#         y += [0.0] # not a match
#         non_matches += 1
#         train_pairs.append((pairs[i][0], f'R{torch.randint(0, int(1e6), (1,))[0]:06d}'))
#     else:
#         if matches == num_matches:
#             continue
#         y += [1.0] # match
#         matches += 1
#         train_pairs.append(pairs[i])
for i in range(len(pairs)):
    if pairs[i][2] == '0':
        if non_matches == num_matches: # Work with subset such that both classes are equally distributed
            continue
        y += [0.0] # not a match
        non_matches += 1
        train_pairs.append(pairs[i][:2])
    else:
        if matches == num_matches:
            continue
        y += [1.0] # match
        matches += 1
        train_pairs.append(pairs[i][:2])

    
if args.basenet[:6] == 'resnet':
    t = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224), # This needs to change
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
else:
    t = transforms.Compose([
        transforms.Resize(128),
        # transforms.CenterCrop(128), # This needs to change
        transforms.RandomResizedCrop(128),
        transforms.ToTensor()
    ])

dataset = ImageDataset(train_pairs, y, transforms=t)
N = len(train_pairs)
print(N)
trainset, testset, valset = random_split(dataset, [int(N*0.7), int(N*0.15), N-(int(N*0.15)+int(N*0.7))])

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
model = MatchNet('learnt').to(device)
loss_fn = nn.BCELoss() # TODO: Needs to be improved
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2w)
train_step = make_train_step(model, loss_fn, optimizer)

if args.load_model:
    directory = args.saveroot+'/'+'models/'
    spath=newest(directory)
    dccheckpoint = torch.load(spath[0])
    model.load_state_dict(dccheckpoint['allstate_dict'])


n_epochs = args.epochs
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

    validation_acc = []
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
            predictions = yhat > 0.5 # Arbitrary threshold
            validation_acc.extend([i.item() for i in predictions == y])
        validation_acc = np.array(validation_acc).mean().item()
        validation_losses.append(np.array(valloss).mean())
    
    print(f"EPOCH {epoch}: Train loss: {training_losses[-1]:0.5f} \t Val loss: {validation_losses[-1]:0.5f} \t Val acc: {validation_acc:0.5f}")

    if epoch % 5 == 0 or epoch == n_epochs-1:
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
        ax.set_ylim(bottom=0)
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

# # Actual evaluation invloves attaching each query in our query set with each ref from 1M images
# ref_ids = []
# with open(args.refset, 'r') as csvfile: 
#     csvreader = csv.reader(csvfile) 
#     fields = next(csvreader)
#     for row in csvreader: 
#         ref_ids.append(row[0])
# query_ids = []
# with open(args.testqset, 'r') as csvfile: 
#     csvreader = csv.reader(csvfile) 
#     fields = next(csvreader)
#     for row in csvreader: 
#         query_ids.append(row[0])

# pairs = []
# for qid in query_ids:
#     for rid in ref_ids:
#         pairs.append([qid, rid])
# test_dataset = ImageDataset(pairs, y=None, transforms=t)
# testloader = DataLoader(test_dataset,
#                         batch_size=args.batch_size,
#                         shuffle=False,
#                         drop_last=False,
#                         num_workers=args.num_workers)  
#### TODO: We also need to keep track of qid, rid. Not just the images for this part

# f = open(args.result_file, 'w')
# with torch.no_grad():
#     for f in tqdm(testloader):
#         Q, R = batch
#         Q = Q.to(device)
#         R = R.to(device)
#         model.eval()
#         yhat = model(Q, R)
#         for i in range(Q.size()[0]):
#             qid = 
#             f.write()
        
#     print(f"Test loss: {np.array(testloss).mean():0.5f}")

'''
####### TODO ########
1. Fix train and test sets. Fix the queries we use and the reference images we use for non-matching pairs
2. Write code to evaluate accurately, ie, output in the format required by the eval script
'''