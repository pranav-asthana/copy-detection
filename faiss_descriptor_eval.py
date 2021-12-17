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
from tqdm import tqdm

from dataclasses import astuple, dataclass
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple
from pl_bolts.models.self_supervised import SimCLR

import faiss


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



def get_embeddings(model, query_ids, ref_ids, query_path, ref_path, device):

  q_embeddings = {} 
  r_embeddings = {} 
  i = 0
  j = 0
  with torch.no_grad():
      model.eval()
      for qid in query_ids:
          Q = Image.open(f'{QIMGS}/{qid}.jpg').convert('RGB')
          Q = t(Q)
          Q = Q.view(1, Q.size()[0], Q.size()[1], Q.size()[2])
          Q = Q.to(device)
          qe = model.encoder(Q)
          q_embeddings[qid] = qe
          if(i%500 == 0):
            print("Query Calculation, Step: ", i)
          i = i + 1
          
      for rid in ref_ids:
          R = Image.open(f'{REFIMGS}/{rid}.jpg').convert('RGB')
          R = t(R)
          R = R.view(1, R.size()[0], R.size()[1], R.size()[2])
          R = R.to(device)
          re = model.encoder(R)
          r_embeddings[rid] = re
          if(j%500 == 0):
            print("Reference Calculation, Step: ", j)
          j = j + 1
    
  return q_embeddings, r_embeddings



"""
Some functions below are taken from the official ISC repository for evaluation. 
"""

@dataclass
class GroundTruthMatch:
    query: str
    db: str

@dataclass
class PredictedMatch:
    query: str
    db: str
    score: float
    
@dataclass
class Metrics:
    average_precision: float
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray
    recall_at_p90: float
    threshold_at_p90: float
    recall_at_rank1: float
    recall_at_rank10: float

def check_duplicates(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    """
    Raise an exception if predictions contains duplicates
    (ie several predictions for the same (query, db) pair).
    """
    unique_pairs = set((p.query, p.db) for p in predictions)
    if len(unique_pairs) != len(predictions):
        raise ValueError("Predictions contains duplicates.")


def sanitize_predictions(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    # TODO(lowik) check for other possible loopholes
    check_duplicates(predictions)
    return predictions

def to_arrays(gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]):
    """Convert from list of matches to arrays"""
    predictions = sanitize_predictions(predictions)

    gt_set = {astuple(g) for g in gt_matches}
    probas_pred = np.array([p.score for p in predictions])
    y_true = np.array([(p.query, p.db) in gt_set for p in predictions], dtype=bool)
    return y_true, probas_pred

def precision_recall(
    y_true: np.ndarray, probas_pred: np.ndarray, num_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls and thresholds.
    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.
    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall.
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred

def average_precision(recalls: np.ndarray, precisions: np.ndarray):
    # Order by increasing recall
    # order = np.argsort(recalls)
    # recalls = recalls[order]
    # precisions = precisions[order]

    # Check that it's ordered by increasing recall
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise Exception("recalls array must be sorted before passing in")

    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()
def find_operating_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, required_x: float
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y with x at least `required_x`.
    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = x >= required_x
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]

def find_tp_ranks(gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]):
    q_to_res = defaultdict(list)
    for p in predictions:
        q_to_res[p.query].append(p)
    ranks = []
    not_found = int(1<<35)
    for m in gt_matches:
        if m.query not in q_to_res:
            ranks.append(not_found)
            continue
        res = q_to_res[m.query]
        res = np.array([
            (p.score, m.db == p.db)
            for p in res
        ])
        i, = np.where(res[:, 1] == 1)
        if i.size == 0:
            ranks.append(not_found)
        else:
            i = i[0]
            rank = (res[:, 0] >= res[i, 0]).sum() - 1
            ranks.append(rank)
    return np.array(ranks)


def argsort(seq):
    # from https://stackoverflow.com/a/3382369/3853462
    return sorted(range(len(seq)), key=seq.__getitem__)
def evaluate(
    gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]
) -> Metrics:
    predictions = sanitize_predictions(predictions)
    y_true, probas_pred = to_arrays(gt_matches, predictions)
    p, r, t = precision_recall(y_true, probas_pred, len(gt_matches))
    ap = average_precision(r, p)
    pp90, rp90, tp90 = find_operating_point(p, r, t, required_x=0.9)  # @Precision=90%
    ranks = find_tp_ranks(gt_matches, predictions)
    recall_at_rank1 = (ranks == 0).sum() / ranks.size
    recall_at_rank10 = (ranks < 10).sum() / ranks.size

    return Metrics(
        average_precision=ap,
        precisions=p,
        recalls=r,
        thresholds=t,
        recall_at_p90=rp90,
        threshold_at_p90=tp90,
        recall_at_rank1=recall_at_rank1,
        recall_at_rank10=recall_at_rank10,
    )
def print_metrics(metrics: Metrics):
    print(f"Average Precision: {metrics.average_precision:.5f}")
    if metrics.recall_at_p90 is None:
        print("Does not reach P90")
    else:
        print(f"Recall at P90    : {metrics.recall_at_p90:.5f}")
        print(f"Threshold at P90 : {metrics.threshold_at_p90:g}")
    print(f"Recall at rank 1:  {metrics.recall_at_rank1:.5f}")
    print(f"Recall at rank 10: {metrics.recall_at_rank10:.5f}")


def read_ground_truth(pairs):
    gt_pairs = []
    for pair in pairs:
        gt_pairs.append(GroundTruthMatch(pair[0], pair[1]))
    return gt_pairs


def knn_match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, k, ngpu=-1, metric=faiss.METRIC_L2):

    if faiss.get_num_gpus() == 0 or ngpu == 0:
        D, I = faiss.knn(xq, xb, k, metric)
    else:
        d = xq.shape[1]
        index = faiss.IndexFlat(d, metric)
        index.add(xb)
        index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(xq, k=k)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        D = -D

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[I[i, j]],
            D[i, j]
        )
        for i in range(nq)
        for j in range(k)
    ]
    return predictions



def evaluate_predictions(ref_embeddings, query_embeddings, k_nn, n, pairs, method='simclr'):
  ip = False
  d = 2048
  gt = read_ground_truth(pairs)
  xb = np.zeros((n,d)).astype('float32')
  xq = np.zeros((n,d)).astype('float32')

  i = 0
  for rid, rembed in ref_embeddings.items():
    if(method == 'simclr'):
      rembed = rembed[0]
    xb[i] = rembed.to("cpu")
    i = i + 1

  i = 0
  for qid, qembed in query_embeddings.items():
    if(method == 'simclr'):
      qembed = qembed[0]
    xq[i] = qembed.to("cpu")
    i = i + 1

  faiss.normalize_L2(xb)
  faiss.normalize_L2(xq)
  predictions = knn_match_and_make_predictions(
  xq, query_ids,
  xb, ref_ids,
  k_nn,
  metric = faiss.METRIC_INNER_PRODUCT if ip else faiss.METRIC_L2
)

  print(f"Evaluating {len(predictions)} predictions ({len(gt)} GT matches)")

  metrics = evaluate(gt, predictions)
  print_metrics(metrics)



### Custom function for binary accuracy

def evaluate_embeddings(q_embeddings, r_embeddings, n=5685, k=5,d =2048, method='custom'):

  # Build FAISS index on reference embeddings

  #d = 2048#q_embeddings[next(iter(q_embeddings.keys()))][0].shape[1]             # dimension
  print(n,d)

  xb = np.zeros((n,d)).astype('float32')
  xq = np.zeros((n,d)).astype('float32')

  i = 0
  for rid, rembed in r_embeddings.items():
    if(method == 'simclr'):
      rembed = rembed[0]
    xb[i] = rembed.to("cpu")
    i = i + 1

  i = 0
  for qid, qembed in q_embeddings.items():
    if(method == 'simclr'):
      qembed = qembed[0]
    xq[i] = qembed.to("cpu")
    i = i + 1


  index = faiss.IndexFlatL2(d)   # build the index
  print("Building index....")
  print("Built index, status: ", index.is_trained)
  index.add(xb)                  # add vectors to the index
  print("Total elements in index: ", index.ntotal)

  D, I = index.search(xq, k)     # actual search
  #print(I[:5])                   # neighbors of the 5 first queries
  #print(I[-5:])                  # neighbors of the 5 last queries

  ct = 0
  for i in range(n):
    if(np.where(I[i] == i)[0].size != 0):
      ct = ct + 1
  print("Pairs found for method {} amongst {} nearest matches: {}".format(method, k,ct))
  return xb, xq


device = "cuda"

def read_mapping_onlymatch(fname):
    pairs = []
    with open(fname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        for row in csvreader: 
            #if(int(row[2:3][0]) == 1):
            pairs.append(row[:2])
    return pairs



pairs = read_mapping_onlymatch(args.testset)

ref_ids = []
query_ids = []
QIMGS = args.qimgs
REFIMGS = args.refimgs 

for pair in pairs:
  query_ids.append(pair[0])
  ref_ids.append(pair[1])


# load resnet50 pretrained using SimCLR on imagenet
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
simclr = simclr.to("cuda")
simclr_resnet50 = simclr.encoder


# Load our Model 

model = MatchNet('learnt').to("cuda")
checkpoint = torch.load('checkpoint-35.pth.tar') ## TODO edit this to path of best model on cluster
model.load_state_dict(checkpoint['allstate_dict'])

# This model is our model's encoder
q_embeddings, r_embeddngs = get_embeddings(model, query_ids, ref_ids, QIMGS, REFIMGS, "cuda" )
# This is SimCLR
q_embed_simclr, r_embed_simclr = get_embeddings(simclr, query_ids, ref_ids, QIMGS, REFIMGS, "cuda" )

# Note that k = 10
#xb_simclr, xq_simclr = evaluate_embeddings(q_embed_simclr, r_embed_simclr, n=len(query_ids),k=10,method='simclr')
#xb_custom, xq_custom = evaluate_embeddings(q_embeddings, r_embeddings, n=len(query_ids),k=10,method='custom')

n = len(query_ids)
evaluate_predictions(r_embeddings, q_embeddings,10, n, pairs,  method='custom')
evaluate_predictions(r_embed_simclr, q_embed_simclr, 10, n, pairs,  method='simclr')


