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
import sys
import argparse
import csv
from PIL import Image
import pickle as pkl
from tqdm import tqdm

'''
 TODO
 1. Experiment with different number of layers frozen
 2. Experiment with different base nets
 3. Experiment with different LR, batch size, etc
'''


parser = argparse.ArgumentParser()
parser.add_argument('--imsuffix', default='.jpg', type=str)
parser.add_argument('--trainset', default='/home/pka4/CV/copy-detection/trainset.csv', type=str)
parser.add_argument('--valset', default='/home/pka4/CV/copy-detection/valset.csv', type=str)
parser.add_argument('--testset', default='/home/pka4/CV/copy-detection/testset.csv', type=str)
parser.add_argument('--refset', default='/home/pka4/CV/copy-detection/refset.csv', type=str)
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
parser.add_argument('--load-model', action='store_true', default=False)
parser.add_argument('--eval', action='store_true',  default=False)
parser.add_argument('--matches', default=4991, type=int)
parser.add_argument('--load-checkpoint', default=-1, type=int)
args, _ = parser.parse_known_args()

device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
print(device)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, transforms=None):
        self.pairs = pairs
        self.labels = labels
        # if labels == None:
        #     self.labels = np.zeros((len(pairs), ))
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
    if args.load_checkpoint == -1:
        return sorted(paths, key=os.path.getctime, reverse=True)[0]
    else:
        return os.path.join(path, f'checkpoint-{args.load_checkpoint}.pth.tar')

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


# num_matches = 4991 # Known, but we can use a subset of known matches too
# num_matches = args.matches
# non_matches, matches = 0, 0
# train_pairs = []
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
# for i in range(len(pairs)):
#     if pairs[i][2] == '0':
#         if non_matches == num_matches: # Work with subset such that both classes are equally distributed
#             continue
#         y += [0.0] # not a match
#         non_matches += 1
#         train_pairs.append(pairs[i][:2])
#     else:
#         if matches == num_matches:
#             continue
#         y += [1.0] # match
#         matches += 1
#         train_pairs.append(pairs[i][:2])

if args.basenet[:6] == 'resnet':
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224), # This needs to change
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(10),
        # transforms.RandomCrop(224, padding=16, padding_mode='reflect'),
        transforms.ToTensor()
    ])
else:
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(128), # This needs to change
        # transforms.RandomResizedCrop(128),
        transforms.ToTensor()
    ])

pairs = read_mapping(args.trainset)
qr = np.array(pairs)[:, :2]
y = np.array(pairs)[:, 2].astype(np.float32)
trainset = ImageDataset(qr, y, transforms=t)
trainloader = DataLoader(trainset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)
print(f'{len(pairs)} train pairs')

test_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

pairs = read_mapping(args.valset)
qr = np.array(pairs)[:, :2]
y = np.array(pairs)[:, 2].astype(np.float32)
valset = ImageDataset(qr, y, transforms=test_t)
valloader = DataLoader(valset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)
print(f'{len(pairs)} val pairs')



pairs = read_mapping(args.testset)
qr = np.array(pairs)[:, :2]
y = np.array(pairs)[:, 2].astype(np.float32)
testset = ImageDataset(qr, y, transforms=test_t)
testloader = DataLoader(testset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)                                                                
print(f'{len(pairs)} test pairs')

torch.manual_seed(42)
model = MatchNet('learnt').to(device)
loss_fn = nn.BCELoss() # TODO: Needs to be improved
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2w)
train_step = make_train_step(model, loss_fn, optimizer)

if args.load_model or args.eval:
    directory = args.saveroot+'/'+'models/'
    spath=newest(directory)
    dccheckpoint = torch.load(spath)
    model.load_state_dict(dccheckpoint['allstate_dict'])

n_epochs = args.epochs
training_losses = []
validation_losses = []

if not args.eval:
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
    
# print("Testing")
# test_acc = []
# pred_proba = []
# gt_y = []
# with torch.no_grad():
#     testloss = []
#     for batch, y in tqdm(testloader):
#         gt_y.extend(y.cpu())
#         Q, R = batch
#         Q = Q.to(device)
#         R = R.to(device)
#         y = y.to(device)
#         model.eval()
#         yhat = model(Q, R)
#         pred_proba.extend(yhat.cpu())
#         loss = loss_fn(yhat, y).item()
#         testloss.append(loss)
#         predictions = yhat > 0.5 # Arbitrary threshold
#         test_acc.extend([i.item() for i in predictions == y])
#     test_acc = np.array(test_acc).mean().item()
#     print(f"Test loss: {np.array(testloss).mean():0.5f}\t Test acc: {test_acc:0.5f}")

# from sklearn.metrics import roc_curve, roc_auc_score
# fpr, tpr, thresholds = roc_curve(gt_y, pred_proba)
# test_auc = roc_auc_score(gt_y, pred_proba)
# plt.figure(0).clf()
# plt.plot(fpr,tpr,label=f"Ours - AUC: {test_auc:.5f}")
# plt.plot([0, 1],[0, 1],label=f"Random choice")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"Performance on binary classification (Test set)")
# plt.legend(loc=0)
# plt.savefig(f"{args.saveroot}/roc.png")
# print("Done")


# Actual evaluation invloves attaching each query in our query set with each ref from 1M images
if not args.eval:
    sys.exit(0)

print("Actual Evaluation (SLOW!)")
print("Reading refids")
# ref_ids = []
# with open(args.refset, 'r') as csvfile: 
#     csvreader = csv.reader(csvfile) 
#     fields = next(csvreader)
#     for row in csvreader: 
#         ref_ids.append(row[0])
# ref_ids = ref_ids[:4000]

# Just use fixed set, same as test set ref ids
ref_ids = []
with open(args.testset, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    fields = next(csvreader)
    for row in csvreader: 
        ref_ids.append(row[1])

print("Reading qids")
query_ids = []
with open(args.testset, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    fields = next(csvreader)
    for row in csvreader: 
        query_ids.append(row[0])

predictions = {} # q:[(r1, c1), ..(r10, c10)]
q_embeddings = {} # qid: embedding
r_embeddings = {} # rid: embedding
# query_ids = query_ids[:200]
# ref_ids = ref_ids[:200]
if os.path.exists(f'{args.saveroot}/predictions.pkl'):
    with open(f'{args.saveroot}/predictions.pkl', 'rb') as f:
        predictions = pkl.load(f)
else:
    with torch.no_grad():
        model.eval()
        for qid in tqdm(query_ids):
            Q = Image.open(f'{args.qimgs}/{qid}{args.imsuffix}').convert('RGB')
            Q = test_t(Q)
            Q = Q.view(1, Q.size()[0], Q.size()[1], Q.size()[2])
            Q = Q.to(device)
            qe = model.encoder(Q)
            q_embeddings[qid] = qe
            
        for rid in tqdm(ref_ids):
            R = Image.open(f'{args.refimgs}/{rid}{args.imsuffix}').convert('RGB')
            R = test_t(R)
            R = R.view(1, R.size()[0], R.size()[1], R.size()[2])
            R = R.to(device)
            re = model.encoder(R)
            r_embeddings[rid] = re
        
        for qid in tqdm(query_ids):
            for rid in (ref_ids):
                yhat = model.distance(q_embeddings[qid], r_embeddings[rid]).cpu().item()
                predictions[qid] = predictions.get(qid, []) + [(rid, yhat)]

    with open(f'{args.saveroot}/predictions.pkl', 'wb') as f:
        pkl.dump(predictions, f)

def draw_below_img(img, text):
    from PIL import ImageFont, ImageDraw, ImageOps
    img = transforms.ToPILImage()(img)
    img = ImageOps.expand(img, border=10, fill=(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0,0),text, (0,0,0))
    return transforms.ToTensor()(img)

print("Writing predictions to file")
f = open(f'{args.saveroot}/predictions.csv', 'w')
# No header required I believe
idx=0
total = 0
found = 0
for pair in tqdm(pairs):
    q, rid, y = pair
    predictions[q] = list(set(predictions[q]))
    rids = np.array(predictions[q])[:, 0]
    conf = np.array(predictions[q])[:, 1]
    conf = conf.astype(float) 
    # print(conf.min(), conf.max(), conf.mean(), conf.std(), y)
    # if conf.max() <= 0.945:
    #     # No match. Do not add to the predictions file
    #     # print("NOMATCH")
    #     continue
    # top_pred = np.array(predictions[q])[np.argmax(conf)]
    # f.write(f'{q},{top_pred[0]},{top_pred[1]}\n')

    k=10
    indices = np.argpartition(conf, -k)[-k:]
    topk = np.array(predictions[q])[indices]
    topk = sorted(topk, key=lambda x: x[1], reverse=True)

    if y == '1':
        total += 1
        if rid in np.array(topk)[:, 0]:
            found += 1

    for topi in topk:
        f.write(f'{q},{topi[0]},{topi[1]}\n')
    idx += 1
    if idx > 100:
        continue

    idx += 1

    Q = test_t(Image.open(f'/shared/rsaas/pka4/CV/Project/data/query/{q}.jpg').convert('RGB'))
    # Q = draw_below_img(Q, 'query')
    img = Q.view(1, 3, Q.size()[1], Q.size()[2])
    R = test_t(Image.open(f'/shared/rsaas/pka4/CV/Project/data/ref/{rid}.jpg').convert('RGB'))
    # R = draw_below_img(R, 'reference')
    img = torch.cat([img, R.view(1, 3, R.size()[1], R.size()[2])], dim=0)
    
    for topi in topk:
        R = test_t(Image.open(f'/shared/rsaas/pka4/CV/Project/data/ref/{topi[0]}.jpg').convert('RGB'))
        # R = draw_below_img(R, f'{topi[1]}')
        img = torch.cat([img, R.view(1, 3, R.size()[1], R.size()[2])], dim=0)
    if y == '0':
        save_image(img, f'/home/pka4/CV/copy-detection/test_check/nomatch_{q}.jpg', nrow=12) 
    if y == '1':
        save_image(img, f'/home/pka4/CV/copy-detection/test_check/match_{q}.jpg', nrow=12) 

print(found, total, found/total)

f.close()
