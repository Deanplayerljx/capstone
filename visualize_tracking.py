import cv2
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--track_file', type=str, required=True,
                    help='the fixed lag track result json file')
parser.add_argument('--img_dir', type=str, required=True,
                    help='the path to the image folder')
parser.add_argument('--out_dir', type=str, required=True,
                    help='the folder to save visualization results')
args = parser.parse_args()

lag = 5
iou_thresh = 0.5
mean_iou_thresh = 0.3
with open(args.track_file, 'r') as f:
    track_results = json.load(f)

img_names = sorted(track_results.keys())
num_colors = 200
colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(num_colors)]

def get_intrinsic_m():
    fxs = []
    fys = []
    skews = []
    u0s = []
    v0s = []
    r0s = []
    r1s = []
    r2s = []
    t0s = []
    t1s = []

    '''
    f_x s   u_0
    0   f_y v_0
    0   0   1 
    '''
    with open("Intrinsic_0000.txt") as f:
        count = 0
        for l in f:
            if count %2 == 0:
                fid, _,_,w,h,fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, _, _ = l.strip().split(' ')
                fxs.append(float(fx))
                fys.append(float(fy))
                skews.append(float(skew))
                u0s.append(float(u0))
                v0s.append(float(v0))
                r0s.append(float(r0))
                r1s.append(float(r1))
                r2s.append(float(r2))
                t0s.append(float(t0))
                t1s.append(float(t1))
            count += 1

    fx = np.mean(fxs)
    fy = np.mean(fys)
    skew = np.mean(skews)
    u0 = np.mean(u0s)
    v0 = np.mean(v0s)
    r0 = np.mean(r0s)
    r1 = np.mean(r1s)
    r2 = np.mean(r2s)
    t0 = np.mean(t0s)
    t1 = np.mean(t1s)
    f = (fx+fy) / 2  # we take the average of fx, fy as focal length here
    K = np.array([[fx, skew, u0],[0, fy, v0],[0, 0, 1]])
    return K, r0, r1, r2, t0, t1

def get_extrinsics():
    extrinsics = []
    with open("CamPose_0000.txt") as f:
        count = 0
        for l in f:
            if count % 2 == 0:
                # rotx, roty, rotz, transX, transY, transZ
                extrinsic =  l.strip().split(" ")
                extrinsic = [float(x) for x in extrinsic[1:]]
                extrinsics.append(extrinsic)
            count += 1
    return extrinsics


def get_extrinsic_matrix(rotx, roty, rotz, transX, transY, transZ):
    Rx = np.array([
        [1,0,0],
        [0, np.cos(rotx), -np.sin(rotx)],
        [0, np.sin(rotx), np.cos(rotx)]
    ])
    Ry = np.array([
        [np.cos(roty),0,np.sin(roty)],
        [0, 1, 0],
        [-np.sin(roty), 0, np.cos(roty)]
    ])
    Rz = np.array([
        [np.cos(rotz),-np.sin(rotz),0],
        [np.sin(rotz), np.cos(rotz), 0],
        [0, 0, 1]
    ]) 

    R = Rx@Ry@Rz
    # R = cv2.Rodrigues(np.array([rotx, roty, rotz]))[0]
    t = np.array([transX, transY, transZ]).reshape(-1,1)
    ex_m = np.hstack((R, t))
    return ex_m

def smooth_tracklet(bboxes, sigma):
    c_xs= [(bbox[0]+bbox[2])/2 for bbox in bboxes]
    c_ys = [(bbox[1]+bbox[3])/2 for bbox in bboxes]
    ws = [bbox[2] - bbox[0] for bbox in bboxes]
    hs = [bbox[3] - bbox[1] for bbox in bboxes]
    smoothed = gaussian_filter1d([c_xs, c_ys, ws, hs], axis=1, sigma=sigma)

    x1s_sm = smoothed[0]  - smoothed[2] / 2
    y1s_sm = smoothed[1]  - smoothed[3] / 2
    x2s_sm = smoothed[0]  + smoothed[2] / 2
    y2s_sm = smoothed[1]  + smoothed[3] / 2

    bboxes_sm = np.vstack((x1s_sm, y1s_sm, x2s_sm, y2s_sm)).T.tolist()
    return bboxes_sm

def get_tracklets():
    tracklets_dict = {}

    for fid, img_name in enumerate(img_names):
        dets = track_results[img_name]
        for det in dets:
            id = det['id']
            bbox = det['bbox']
            if id not in tracklets_dict:
                tracklets_dict[id] = [{'fid':fid, 'bbox':bbox}]
            else:
                # we only want continuous tracklets
                if tracklets_dict[id][-1]['fid'] == fid - 1:
                    tracklets_dict[id].append({'fid':fid, 'bbox':bbox})
    print("num tracklets: {}".format(len(tracklets_dict)))
    ids_to_del = []
    for id in tracklets_dict:
        if len(tracklets_dict[id]) == 1:
            ids_to_del.append(id)

    for id in ids_to_del:
        del tracklets_dict[id]

    print("num tracklets: {}".format(len(tracklets_dict)))
    return tracklets_dict

def undistort_tracklet(bboxes, K, dist_coeff, new_K):
    bboxes = np.array(bboxes).reshape((-1, 2))
    bboxes = cv2.undistortPoints(bboxes, K, np.array(dist_coeff), P=new_K)
    bboxes = np.squeeze(bboxes)
    return bboxes.reshape((-1, 4))


def visualize_undistorted_by_id(out_img_dir, ids):
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    K, r0, r1, r2, t0, t1 = get_intrinsic_m()
    dist_coeff = np.array([r0, r1, t0, t1, r2])
    extrinsics = get_extrinsics()
    ex_m_first = get_extrinsic_matrix(*extrinsics[0])
    ex_m_first = np.vstack((ex_m_first, [[0,0,0,1]]))
    ex_m_first_inv = np.linalg.inv(ex_m_first)


    img_size = (1920, 1080)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, img_size, alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, np.eye(3), new_K, img_size, cv2.CV_32FC1)
    print("original K")
    print(K)
    print("new K")
    print(new_K)

    tracklets_dict = get_tracklets()
    tracklet_ids = set(tracklets_dict.keys())

    processed_dets = defaultdict(list)
    for id in ids:
        bboxes = [x['bbox'] for x in tracklets_dict[id]]
        fids = [x['fid'] for x in tracklets_dict[id]]

        bboxes = smooth_tracklet(bboxes, sigma=1)
        bboxes = undistort_tracklet(bboxes, K, dist_coeff, new_K)
        
        for fid, bbox in zip(fids, bboxes):
            processed_dets[fid].append({'bbox':bbox, 'id':id})

    for fid, img_name in enumerate(tqdm(img_names[:30])):
        img = cv2.imread(os.path.join(args.img_dir, img_name))
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        dets = processed_dets[fid]
        for det in dets:
            bbox = [int(x) for x in det['bbox']]
            id = int(det['id'])
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            long = max(h,w)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[id%num_colors], thickness=2)
            cv2.putText(img, str(id), (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=long/200+0.5, color=colors[id%num_colors], thickness=2)
        cv2.imwrite(os.path.join(out_img_dir, img_name), img)


def visualize_original(out_img_dir):
    for img_name in tqdm(img_names[:50]):
        img = cv2.imread(os.path.join(args.img_dir, img_name))
        dets = track_results[img_name]
        for det in dets:
            bbox = [int(x) for x in det['bbox']]
            id = int(det['id'])
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            long = max(h,w)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[id%num_colors], thickness=2)
            cv2.putText(img, str(id), (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=long/200+0.5, color=colors[id%num_colors], thickness=2)
        cv2.imwrite(os.path.join(out_img_dir, img_name), img)

def visualize_original_by_id(out_img_dir, ids):
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    for img_name in tqdm(img_names[:30]):
        img = cv2.imread(os.path.join(args.img_dir, img_name))
        dets = track_results[img_name]
        for det in dets:
            bbox = [int(x) for x in det['bbox']]
            id = int(det['id'])
            if id not in ids:
                continue
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            long = max(h,w)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[id%num_colors], thickness=2)
            cv2.putText(img, str(id), (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=long/200+0.5, color=colors[id%num_colors], thickness=2)
        cv2.imwrite(os.path.join(out_img_dir, img_name), img)

def main():
    all_persons = [1, 322, 491]+[2, 492]+[4, 324, 428] + [69, 624] + [50, 123, 183, 252, 301, 345, 399]
    visualize_original_by_id(args.out_dir, all_persons)
    # visualize_undistorted_by_id('first_30', all_persons)


if __name__ == "__main__":
    main()
