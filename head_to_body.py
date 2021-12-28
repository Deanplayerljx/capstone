import numpy as np
import json
import cvxpy as cp
from scipy.ndimage import gaussian_filter1d
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from util.camera_pose_visualizer import CameraPoseVisualizer
from collections import defaultdict

def project_image_to_rect(K, uv_depth):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    c_u = K[0,2]
    c_v = K[1,2]
    f_u = K[0,0]
    f_v = K[1,1]
    b_x = 0
    b_y = 0
    n = uv_depth.shape[0]
    x = ((uv_depth[:,0]-c_u)*uv_depth[:,2])/f_u + b_x
    y = ((uv_depth[:,1]-c_v)*uv_depth[:,2])/f_v + b_y
    pts_3d_rect = np.zeros((n,3))
    pts_3d_rect[:,0] = x
    pts_3d_rect[:,1] = y
    pts_3d_rect[:,2] = uv_depth[:,2]
    return pts_3d_rect

def project_rect_to_image(K, pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_2d = np.dot(pts_3d_rect, np.transpose(K)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def headbox2human(head, K):
    h = np.array(head).reshape((2, 2))
    
    depth_x = K[0, 0] * 0.152 / (head[2] - head[0])
    depth_y = K[1, 1] * 0.22 / (head[3] - head[1])
    depth = (depth_x + depth_y) / 2
    h_depth = np.concatenate((h, [[depth], [depth]]), axis=1)
    cord = project_image_to_rect(K, h_depth)
    center = cord[:, 0:2].mean(axis=0)
    # x1, x2 = center[0] - 0.22, center[0] + 0.22
    x1, x2 = center[0] - 0.27, center[0] + 0.27
    y1, y2 = cord[0, 1], cord[0, 1] + 1.7
    humanbox_3d = np.array([[x1, y1, depth], [x2, y2, depth]])
    return project_rect_to_image(K, humanbox_3d)

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
            if count %2 == 0: # the parameters were obtained from 60fps video
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
    # f = (fx+fy) / 2  # we take the average of fx, fy as focal length here
    K = np.array([[fx, skew, u0],[0, fy, v0],[0, 0, 1]])
    return K, r0, r1, r2, t0, t1



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

    # multiply the translation term by scale factor in SfT to make it metric
    transX_scaled = transX * SFM_SCALE
    transY_scaled = transY * SFM_SCALE
    transZ_scaled = transZ * SFM_SCALE

    t = np.array([transX_scaled, transY_scaled, transZ_scaled]).reshape(-1,1)

    ex_m = np.hstack((R, t))
    return ex_m

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
    # ids_to_del = []
    # for id in tracklets_dict:
        # if len(tracklets_dict[id]) == 1:
            # ids_to_del.append(id)

    # for id in ids_to_del:
        # del tracklets_dict[id]

    print("num tracklets: {}".format(len(tracklets_dict)))
    return tracklets_dict

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

def undistort_tracklet(bboxes, K, dist_coeff, new_K):
    bboxes = np.array(bboxes).reshape((-1, 2))
    bboxes = cv2.undistortPoints(bboxes, K, np.array(dist_coeff), P=new_K)
    bboxes = np.squeeze(bboxes)
    # print(K)
    # print(bboxes.shape)
    # bboxes[:, 0] *= K[0][0] + K[0][2]
    # bboxes[:, 1] *= K[1][1] + K[1][2]
    return bboxes.reshape((-1, 4))

def draw_body_bboxes(img_dir, out_dir, img2body):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for img_name, dets in img2body.items():
        img = cv2.imread(os.path.join(img_dir,img_name))
        for det in dets:
            bbox = det['body_bbox']
            bbox = [int(x) for x in bbox]

            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            long = max(h,w)

            if not USE_DET:
                id = det['id']
                cv2.putText(img, str(id), (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=w/200+0.5, color=colors[id%num_colors], thickness=2)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colors[id%num_colors], thickness=2)
            else:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,255), thickness=2)

        cv2.imwrite(os.path.join(out_dir, img_name), img)


def traj_head_to_body(id, K, new_K, img_w, img_h, undistort):
    bboxes = [x['bbox'] for x in tracklets_dict[id]]
    fids = [x['fid'] for x in tracklets_dict[id]]

    bboxes = smooth_tracklet(bboxes, sigma=1)

    if undistort:
        # undistort bboxes
        bboxes = undistort_tracklet(bboxes, K, dist_coeff, new_K)

    body_bboxes = []
    for bbox in bboxes:
        if undistort:
            body_bbox = headbox2human(bbox, new_K).flatten()
        else:
            body_bbox = headbox2human(bbox, K).flatten()
        body_bbox[0] = max(0, body_bbox[0])
        body_bbox[1] = max(0, body_bbox[1])
        body_bbox[2] = min(img_w-1, body_bbox[2])
        body_bbox[3] = min(img_h-1, body_bbox[3])
        body_bboxes.append(body_bbox)

    return body_bboxes

track_file_name = 'lag_5_cross_check_0.3_track_results_merged_algo.json' 
with open(track_file_name, 'r') as f:
    track_results = json.load(f)
img_names = sorted(track_results.keys())
# img_names = img_names[:30]


tracklets_dict = get_tracklets()
tracklet_ids = set(tracklets_dict.keys())

num_colors = 200
colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(num_colors)]

K, r0, r1, r2, t0, t1 = get_intrinsic_m()
dist_coeff = np.array([r0, r1, t0, t1, r2])

# get transformation for undistortion
img_size = (1920, 1080)
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, img_size, alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, np.eye(3), new_K, img_size, cv2.CV_32FC1)
print("original K")
print(K)
print("new K")
print(new_K)


ids = [1]
ids = tracklet_ids
img2body = defaultdict(list)
UNDISTORT = False
USE_DET = True
if USE_DET:
    out_filename = 'body_bboxes_from_det_andy_smooth.json'
    DEBUG_FOLDER = "debug_regress_from_det_andy_smooth"
    IMG_FOLDER = 'data/ny_china_town_0350_0420/images/'
else:
    if UNDISTORT:
        out_filename = 'body_bboxes_undistorted.json'
        DEBUG_FOLDER = "debug_regress_undistorted"
        IMG_FOLDER = 'data/ny_china_town_0350_0420/undistorted_images/'
    else:
        out_filename = 'body_bboxes.json'
        DEBUG_FOLDER = "debug_regress"
        IMG_FOLDER = 'data/ny_china_town_0350_0420/images/'

DRAW_BBOXES = True
if not USE_DET:
    for id in ids:
        body_bboxes = traj_head_to_body(id, K, new_K, img_size[0], img_size[1], UNDISTORT)
        fids = [x['fid'] for x in tracklets_dict[id]]
        for fid, bbox in zip(fids, body_bboxes):
            img_name = img_names[fid]
            img2body[img_name].append({'id':id, 'body_bbox':bbox.tolist()})

# regress on original detections
if USE_DET:
    # det_filename = 'data/ny_china_town_0350_0420/overfit_tina_face_0.3_0.45/result_full.json'
    det_filename = '/home/ajwei/tinaface/vedadet/data/face_clip/images/ny_china_town_0350_0420/full_infer_1118_smooth/tina_face_finetune_0.01_0.45/result_full.json'
    with open(det_filename, 'r') as f:
        det_results = json.load(f)

    for img_name in img_names:
        dets = det_results[img_name] 
        body_bboxes = []
        for bbox in dets:
            bbox = bbox[:4]
            body_bbox = headbox2human(bbox, K).flatten()
            body_bbox[0] = max(0, body_bbox[0])
            body_bbox[1] = max(0, body_bbox[1])
            body_bbox[2] = min(img_size[0]-1, body_bbox[2])
            body_bbox[3] = min(img_size[1]-1, body_bbox[3])
            body_bboxes.append(body_bbox)

        for bbox in body_bboxes:
            img2body[img_name].append({'body_bbox': bbox.tolist()})

with open(out_filename, 'w') as f:
    json.dump(img2body,f)

if DRAW_BBOXES:
    draw_body_bboxes(IMG_FOLDER, DEBUG_FOLDER, img2body)

