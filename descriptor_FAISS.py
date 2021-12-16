# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18JwVf-eoiGN8Ey-tB0CcgMfdyhjKiehY
"""

# Commented out IPython magic to ensure Python compatibility.
"""Headers""" 
from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchsummary import summary

from sklearn.metrics import confusion_matrix

import csv
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import astuple, dataclass
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple
from pl_bolts.models.self_supervised import SimCLR

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)

!pip install pytorch-lightning
!pip install lightning-bolts
!pip install faiss
!sudo apt-get install libomp-dev

import faiss
import tqdm

def read_mapping(fname):
    pairs = []
    with open(fname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        for row in csvreader: 
            pairs.append(row)
    return pairs

device = "cuda"
ref_ids = []
query_ids = []
QIMGS = '/data/aug_queries_1' # TODO CHANGE TO PATH ON SYSTEM
REFIMGS = '/data/aug_ref' # TODO CHANGE TO PATH ON SYSTEM

for pair in pairs:
  query_ids.append(pair[0])
  ref_ids.append(pair[1])

def get_embeddings(model, query_ids, ref_ids, query_path, ref_path, device):

  q_embeddings = {} # qid: embedding
  r_embeddings = {} # rid: embedding
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

import numpy as np
import faiss
from faiss.contrib import exhaustive_search
import logging

def query_iterator(xq):
    """ produces batches of progressively increasing sizes """
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i : i + bs]
        yield xqi
        if bs < 20000:
            bs *= 2
        i += len(xqi)

#########################
# These two functions are there because current Faiss contrib
# does not proporly support IP search
#########################


def threshold_radius_nres_IP(nres, dis, ids, thresh):
    """ select a set of results """
    mask = dis > thresh
    new_nres = np.zeros_like(nres)
    o = 0
    for i, nr in enumerate(nres):
        nr = int(nr)   # avoid issues with int64 + uint64
        new_nres[i] = mask[o : o + nr].sum()
        o += nr
    return new_nres, dis[mask], ids[mask]


def apply_maxres_IP(res_batches, target_nres):
    """find radius that reduces number of results to target_nres, and
    applies it in-place to the result batches used in range_search_max_results"""
    alldis = np.hstack([dis for _, dis, _ in res_batches])
    alldis.partition(len(alldis) - target_nres)
    radius = alldis[-target_nres]

    LOG = logging.getLogger(exhaustive_search.__name__)

    if alldis.dtype == 'float32':
        radius = float(radius)
    else:
        radius = int(radius)
    LOG.debug('   setting radius to %s' % radius)
    totres = 0
    for i, (nres, dis, ids) in enumerate(res_batches):
        nres, dis, ids = threshold_radius_nres_IP(nres, dis, ids, radius)
        totres += len(dis)
        res_batches[i] = nres, dis, ids
    LOG.debug('   updated previous results, new nb results %d' % totres)
    return radius, totres

def search_with_capped_res(xq, xb, num_results, metric=faiss.METRIC_L2):
    """
    Searches xq into xb, with a maximum total number of results
    """
    index = faiss.IndexFlat(xb.shape[1], metric)
    index.add(xb)
    # logging.basicConfig()
    # logging.getLogger(exhaustive_search.__name__).setLevel(logging.DEBUG)

    if metric == faiss.METRIC_INNER_PRODUCT:
        # this is a very ugly hack because contrib.exhaustive_search does
        # not support IP search correctly. Do not use in a multithreaded env.
        apply_maxres_saved = exhaustive_search.apply_maxres
        exhaustive_search.apply_maxres = apply_maxres_IP

    radius, lims, dis, ids = exhaustive_search.range_search_max_results(
        index, query_iterator(xq),
        1e10 if metric == faiss.METRIC_L2 else -1e10,      # initial radius does not filter anything
        max_results=2 * num_results,
        min_results=num_results,
        ngpu=-1   # use GPU if available
    )

    if metric == faiss.METRIC_INNER_PRODUCT:
        exhaustive_search.apply_maxres = apply_maxres_saved

    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        if metric == faiss.METRIC_L2:
            o = dis.argpartition(num_results)[:num_results]
        else:
            o = dis.argpartition(len(dis) - num_results)[-num_results:]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [
            mask[lims[i] : lims[i + 1]].sum()
            for i in range(nq)
        ]
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids


def match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, num_results, ngpu=-1, metric=faiss.METRIC_L2):
    lims, dis, ids = search_with_capped_res(xq, xb, num_results, metric=metric)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        dis = -dis

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[ids[j]],
            dis[j]
        )
        for i in range(nq)
        for j in range(lims[i], lims[i + 1])
    ]
    return predictions

def evaluate_predictions(ref_embeddings, query_embeddings, method='simclr'):
  k_nn = 5 # Numberof nearest neighbors 
  ip = False
  gt_filepath = './augmented_ground_truth_1.csv' # TODO Relace
  n = 5685 # TODO Replace
  d = 2048
  gt = read_ground_truth(gt_filepath)
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

  faiss.normalize_L2(xb)
  faiss.normalize_L2(xq)
  predictions = knn_match_and_make_predictions(
  xq, query_ids,
  xb, ref_ids,
  k_nn,
  metric = faiss.METRIC_INNER_PRODUCT if ip else faiss.METRIC_L2
)
  # predictions = match_and_make_predictions(
  #         xq, query_ids,
  #         xb, ref_ids,
  #         10000,
  #         metric = faiss.METRIC_INNER_PRODUCT if ip else faiss.METRIC_L2
  #     )

  print(f"Evaluating {len(predictions)} predictions ({len(gt)} GT matches)")

  metrics = evaluate(gt, predictions)
  print_metrics(metrics)



def read_ground_truth(filename: str) -> List[GroundTruthMatch]:
    """
    Read groundtruth csv file.
    Must contain query_image_id,db_image_id on each line.
    handles the no header version and DD's version with header
    """
    gt_pairs = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == 'query_id,reference_id':
                continue
            q, db = line.split(",")
            if db == '':
                continue
            gt_pairs.append(GroundTruthMatch(q, db))
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



# load resnet50 pretrained using SimCLR on imagenet
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
simclr = simclr.to("cuda")
simclr_resnet50 = simclr.encoder


# This model is our model's encoder
q_embeddings, r_embeddngs = get_embeddings(model, query_ids, ref_ids, QIMGS, REFIMGS, "cuda" )
# This is SimCLR
q_embed_simclr, r_embed_simclr = get_embeddings(simclr, query_ids, ref_ids, QIMGS, REFIMGS, "cuda" )

# Note that k = 10
xb_simclr, xq_simclr = evaluate_embeddings(q_embed_simclr, r_embed_simclr, n=len(query_ids),k=10,method='simclr')
xb_custom, xq_custom = evaluate_embeddings(q_embeddings, r_embeddings, n=len(query_ids),k=10,method='custom')