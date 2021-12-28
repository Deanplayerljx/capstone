import argparse
# import imutils
import time
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from cv_utils import iou, tracklet_overlap_iou, tracklet_nms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
import torchvision.ops as ops
import torch

lag = 5
mean_iou_thresh = 0.3
track_results_filename = 'lag_{}_cross_check_track_results.json'.format(lag)
with open(track_results_filename, 'r') as f:
    track_results = json.load(f)

img_dir = 'data/ny_china_town_0350_0420/images'
# recover tracklets from 
img_names = sorted(track_results.keys())
cur_tracklet_pool = set()
merged_track_results = {}
merge_id_map = {}

def get_orig_tracklets(cur_ids, cur_fid):
    id_set = set(cur_ids)
    tracklets = defaultdict(dict)
    tracklets = {}
    for fid in range(max(0, cur_fid-lag*2), cur_fid+1):
        img_name = img_names[fid]
        dets = track_results[img_name]
        for det in dets:
            if det['id'] in id_set:
                if det['id'] not in tracklets:
                    tracklets[det['id']] = {'bboxes': [], 'conf': det['conf']}
                tracklets[det['id']]['bboxes'].append(det['bbox'])
    return tracklets

def merge_tracklets(t1, t2):
    """
    t1: ********
    t2:     ****  
    """
    min_len = min(len(t1), len(t2))
    t1_overlap = np.array(t1[len(t1)-min_len:])
    t2_overlap = np.array(t2[len(t2)-min_len:])
    overlap = ((t1_overlap + t2_overlap) / 2 ).tolist()

    if min_len == len(t1):
        prev_overlap = t2[:len(t2)-min_len] 
    else:
        prev_overlap = t1[:len(t1)-min_len] 

    return prev_overlap + overlap

prev_tracklets_dict = {}
num_colors = 200
colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(num_colors)]
new_track_results = {}
for fid, img_name in enumerate(tqdm(img_names)):
    dets = track_results[img_name].copy()
    ids = np.array([det['id'] for det in dets])

    orig_tracklets_dict = get_orig_tracklets(ids, fid)
    orig_ids = sorted(list(orig_tracklets_dict.keys()))
    #NOTE: the orig_tracklets end at the current timestamp
    orig_tracklets = [orig_tracklets_dict[id]['bboxes'] for id in orig_ids] 
    orig_confs = [orig_tracklets_dict[id]['conf'] for id in orig_ids] 

    if len(prev_tracklets_dict): 
        # Hungarian matching with previous tracklets
        prev_ids = sorted(list(prev_tracklets_dict.keys()))
        prev_tracklets = [prev_tracklets_dict[id]['bboxes'] for id in prev_ids]
        prev_confs = [prev_tracklets_dict[id]['conf'] for id in prev_ids]

        cost_matrix = [] # [num_orig_tracklets, num_prev_tracklets]
        for orig_seq_idx, orig_seq in enumerate(orig_tracklets):
            mean_ious = []
            for prev_seq_idx, prev_seq in enumerate(prev_tracklets):
                mean_iou = tracklet_overlap_iou(orig_seq[:-1], prev_seq) 
                if mean_iou == 1:
                    if orig_ids[orig_seq_idx] == prev_ids[prev_seq_idx]: # give tracklets with same id more weights when tracklets compeletely overlap
                        mean_iou = 200
                    else:
                        mean_iou = 100
                mean_ious.append(mean_iou)
            # mean_ious = [tracklet_overlap_iou(orig_seq[:-1], prev_seq) for prev_seq in prev_tracklets]
            cost_matrix.append(mean_ious)

        cost_matrix = np.array(cost_matrix)
        assigned_orig_indices, assigned_prev_indices = linear_sum_assignment(
            cost_matrix, True)


        """
        unssigned: keep id
        assigned: merge to old id

        goal: have tracklets that end at current timestamp
        """
        cur_tracklets_dict = {}
        assigned_orig_ids = set()
        # for matched tracklets, merge prev tracklet and orig tracklet, use prev id 
        for orig_idx, prev_idx in zip(assigned_orig_indices, assigned_prev_indices):
            if cost_matrix[orig_idx][prev_idx] >= mean_iou_thresh:
                prev_id = prev_ids[prev_idx]
                orig_id = orig_ids[orig_idx]
                assigned_orig_ids.add(orig_id)
                prev_tracklet = prev_tracklets[prev_idx]
                orig_tracklet = orig_tracklets[orig_idx]
                orig_tracklet_prev = orig_tracklet[:-1] # orig tracklet that ended at t-1
                merged_tracklet = merge_tracklets(prev_tracklet, orig_tracklet_prev)
                merged_tracklet += [orig_tracklet[-1]]

                assert(prev_id not in cur_tracklets_dict)
                cur_tracklets_dict[prev_id] = {}
                cur_tracklets_dict[prev_id]['bboxes'] = merged_tracklet
                max_conf = max(prev_confs[prev_idx], orig_confs[orig_idx])
                cur_tracklets_dict[prev_id]['conf'] = max_conf


        # for unmatched orig tracklets, copy the current frame 
        # to the current tracklets
        for orig_idx in range(len(orig_ids)):
            orig_id = orig_ids[orig_idx]
            if orig_id not in assigned_orig_ids:
                # if img_name == '000158.jpg':
                try:
                    assert(orig_id not in cur_tracklets_dict)
                except:
                    print(img_name)
                    print('id is {}'.format(orig_id))
                    prev_idx = prev_ids.index(orig_id)
                    print(np.array(cost_matrix)[:, prev_idx])
                    print(cost_matrix[orig_idx][prev_idx])
                assert(orig_id not in cur_tracklets_dict)

                cur_tracklets_dict[orig_id] = {}
                cur_tracklets_dict[orig_id]['bboxes'] = [orig_tracklets[orig_idx][-1]]
                cur_tracklets_dict[orig_id]['conf'] = orig_confs[orig_idx]

    else:
        cur_tracklets_dict = orig_tracklets_dict


    merged_tracklets_dict = {}
    cur_ids = sorted(list(cur_tracklets_dict.keys()))
    cur_tracklets = [cur_tracklets_dict[id]['bboxes'] for id in cur_ids]
    cur_confs = [cur_tracklets_dict[id]['conf'] for id in cur_ids]

    keep_indices = tracklet_nms(np.array(cur_tracklets, dtype=object), np.array(cur_confs), mean_iou_thresh)
    for keep_idx in keep_indices:
        if cur_ids[keep_idx] not in merged_tracklets_dict:
            merged_tracklets_dict[cur_ids[keep_idx]] = {}
        merged_tracklets_dict[cur_ids[keep_idx]]['bboxes'] = cur_tracklets[keep_idx]
        merged_tracklets_dict[cur_ids[keep_idx]]['conf'] = cur_confs[keep_idx]

    prev_tracklets_dict = merged_tracklets_dict

    # prepare final results to dump
    cur_new_track_result = []
    for id in merged_tracklets_dict:
        cur_new_track_result.append({
            'id': id,
            'bbox':merged_tracklets_dict[id]['bboxes'][-1],
            'conf':merged_tracklets_dict[id]['conf']
        })
    new_track_results[img_name] = cur_new_track_result

out_path = 'lag_{}_cross_check_{}_track_results_merged_algo.json'.format(lag, mean_iou_thresh)
with open(out_path, 'w') as f:
    json.dump(new_track_results, f)