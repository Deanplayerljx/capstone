# from imutils.video import VideoStream
# from imutils.video import FPS
import argparse
# import imutils
import time
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from cv_utils import iou
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--det_file', type=str, required=True,
                    help='the detection result json file')
parser.add_argument('--img_dir', type=str, required=True,
                    help='the path to the image folder')
parser.add_argument('--out_file', type=str, required=True,
                    help='the fixed lag tracking result file')
args = parser.parse_args()

with open(args.det_file, 'r') as f:
    det_results = json.load(f)

trackers = {}
tracker_id = 0
tracklets = {}
tracker_start_time = {}
fid_to_est_bboxes = defaultdict(list) 
track_results = defaultdict(list)

seq_len = 898
iou_thresh = 0.2
rescale = 1
lag = 5
img_names = sorted(det_results.keys())[:seq_len]

def get_unaligned_bboxes(bboxes, confs, est_bboxes):
    cost_matrix = [] # [num_dets, num_ests]
    for bbox_idx, bbox in enumerate(bboxes):
        ious = [iou(bbox, est_bbox) for est_bbox in est_bboxes]
        cost_matrix.append(ious)

    assigned_bbox_indices, assigned_est_indices = linear_sum_assignment(
        cost_matrix, True)


    # find detections that cannot be associated
    unassigned_bbox_indices = set(list(range(len(bboxes)))) - \
        set(assigned_bbox_indices)

    low_iou_bbox_indices = set()
    for r_idx, c_idx in zip(assigned_bbox_indices, assigned_est_indices):
        if cost_matrix[r_idx][c_idx] < iou_thresh:
            low_iou_bbox_indices.add(r_idx)

    new_det_bbox_indices = unassigned_bbox_indices.union(low_iou_bbox_indices)
    return [bboxes[i] for i in new_det_bbox_indices], [confs[i] for i in new_det_bbox_indices]

def get_forward_backward_imgs_from_middle(mid_fid, fid):
    imgs_f = []
    for fid_f in range(mid_fid+1, fid+1):
        img_name = img_names[fid_f]     
        img_path = os.path.join(args.img_dir, img_name)
        img_f = cv2.imread(img_path)
        imgs_f.append(img_f)

    imgs_b = []
    for fid_b in range(mid_fid-1, max(0,mid_fid-lag)-1, -1):
        img_name = img_names[fid_b]     
        img_path = os.path.join(args.img_dir, img_name)
        img_b = cv2.imread(img_path)
        imgs_b.append(img_b)
    return imgs_f, imgs_b

def two_dir_track(tracker, imgs, init_bbox, init_img, debug=False):
    if not imgs: # the current window to track is empty (happens at the start of the sequence)
        return []
    if debug:
        if not os.path.exists('./debug'):
            os.mkdir("./debug")

    init_copy = init_img.copy()
    if debug:
        cv2.rectangle(init_copy,(init_bbox[0], init_bbox[1]), (init_bbox[2], init_bbox[3]), (0,255,0), 2)
        cv2.imwrite("./debug/init.jpg", init_copy)
    last_success_idx = None
    forward_est_bboxes = []
    for idx, img in enumerate(imgs):
        # fid_f = mid_fid + 1 + idx
        success, est_bbox = tracker.update(img)
        if success:
            x1, y1, w, h = est_bbox
            est_bbox_area = w*h
            if est_bbox_area > 1:
                forward_est_bboxes.append([x1, y1, x1+w, y1+h])
                # fid_to_est_bboxes[fid_f].append([x1, y1, x1+w, y1+h])
                last_success_idx = idx
                if debug:
                    img_copy = img.copy()
                    cv2.rectangle(img_copy,(x1, y1), (x1+w, y1+h), (0,255,0), 2)
                    cv2.imwrite("./debug/forward_{}.jpg".format(idx), img_copy)
        else:
            break

    if last_success_idx == None:
        return None

    
    backward_imgs = imgs[:last_success_idx][::-1]
    for idx, img in enumerate(backward_imgs):
        success, est_bbox = tracker.update(img)
        x1, y1, w, h = est_bbox
        if not success:
            # print('failed on backward')
            return None
        else:
            if debug:
                img_copy = img.copy()
                cv2.rectangle(img_copy,(x1, y1), (x1+w, y1+h), (0,255,0), 2)
                cv2.imwrite("./debug/backward_{}.jpg".format(last_success_idx-idx-1), img_copy)

    success, est_bbox = tracker.update(init_img)
    x1, y1, w, h = est_bbox
    if not success: # failed on init image
        # print('failed on init image')
        if debug:
            img_copy = init_img.copy()
            cv2.rectangle(img_copy,(x1, y1), (x1+w, y1+h), (0,255,0), 2)
            cv2.imwrite("./debug/backward_init.jpg", img_copy)
            exit()

        return None
    else:
        if debug:
            img_copy = init_img.copy()
            cv2.rectangle(img_copy,(x1, y1), (x1+w, y1+h), (0,255,0), 2)
            cv2.imwrite("./debug/backward_init.jpg", img_copy)

    x1, y1, w, h = est_bbox
    if iou(init_bbox, [x1, y1, x1+w, y1+h]) < iou_thresh:
        # print('failed on iou check')
        # print(iou(init_bbox, [x1, y1, x1+w, y1+h]))
        return None
    # print(forward_est_bboxes)
    return forward_est_bboxes

    


for fid in tqdm(range(lag, seq_len)):
    mid_fid = fid - lag
    img_name = img_names[mid_fid]
    dets = np.array(det_results[img_name])
    bboxes = (dets[:, :4]*rescale).astype(int)
    areas = np.array([(x[2]-x[0])*(x[3]-x[1]) for x in bboxes])
    bboxes = bboxes[areas > 1]

    confs = dets[:, 4]
    img_path = os.path.join(args.img_dir, img_name)
    img = cv2.imread(img_path)
    h, w, c = img.shape
    img = cv2.resize(img, (int(w*rescale), int(h*rescale)))

    # get unaligned dets 
    est_bboxes = fid_to_est_bboxes[mid_fid]
    if len(est_bboxes) == 0:
        # print('no estimated bbox at frame {}'.format(mid_fid))
        new_det_bboxes = bboxes 
        new_det_confs = confs
    else:
        new_det_bboxes, new_det_confs = get_unaligned_bboxes(bboxes, confs, est_bboxes)
        # print('# unmatched dets: {}'.format(len(new_det_bboxes)))

    # initiate tracking in both directions for unaligned dets
    imgs_f, imgs_b = get_forward_backward_imgs_from_middle(mid_fid, fid)
    for new_det_bbox, new_det_conf in zip(new_det_bboxes, new_det_confs):
        tracker_id += 1
        # NOTE: we do not add new_det_bbox to the fid_to_est_bbox_dict
        track_results[img_names[mid_fid]].append({
            'bbox': new_det_bbox,
            'id': tracker_id,
            'conf': new_det_conf})

        tracker_f = cv2.TrackerKCF_create()
        tracker_b = cv2.TrackerKCF_create()
        x1, y1, x2, y2 = new_det_bbox
        w = x2 - x1
        h = y2 - y1
        tracker_b.init(img, [x1, y1, w, h])
        tracker_f.init(img, [x1, y1, w, h])

        forward_est_bboxes = two_dir_track(tracker_f, imgs_f, [x1, y1, x2, y2], img)
        backward_est_bboxes = two_dir_track(tracker_b, imgs_b, [x1, y1, x2, y2], img)
        if forward_est_bboxes == None or backward_est_bboxes == None:
            continue

        for idx, est_bbox in enumerate(forward_est_bboxes):
            fid_f = mid_fid + 1 + idx
            fid_to_est_bboxes[fid_f].append(est_bbox)
            track_results[img_names[fid_f]].append({
                'bbox': est_bbox,
                'id': tracker_id,
                'conf': new_det_conf})

        for idx, est_bbox in enumerate(backward_est_bboxes):
            fid_b = mid_fid - 1 - idx
            fid_to_est_bboxes[fid_b].append(est_bbox)
            track_results[img_names[fid_b]].append({
                'bbox': est_bbox,
                'id': tracker_id,
                'conf': new_det_conf})

for img_name, dets in track_results.items():
    for det in dets:
        det['bbox'] = (np.array(det['bbox']) / rescale).tolist()
for tid in tracklets:
    start_fid = tracker_start_time[tid]
    tracklet = tracklets[tid] 
    for fid in range(start_fid, start_fid + len(tracklet)):
        img_name = img_names[fid]
        track_results[img_name].append({
            'bbox':(np.array(tracklet[fid-start_fid]) / rescale).tolist(), 
            'id': tid})

with open(args.out_file, 'w') as f:
    json.dump(track_results, f)