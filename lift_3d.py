import json
import numpy as np
import cvxpy as cp
from scipy.ndimage import gaussian_filter1d
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from util.camera_pose_visualizer import CameraPoseVisualizer

DEBUG = False
HEAD_W = 148.0 # human head width (mm)
HEAD_H = 225.0 # human head height (mm)
SFM_SCALE = 1072
# SFM_SCALE = 1

# Controls which method to use for making all lifted points 
# relative to the first camera coordinates:
# 1. first multiply all camera extrinsics by the inverse of the 
# first camera's intrinsic matrix, then lift points to 3d
# 2. first lift points to 3d, then transform all points by the 
# the first camera's extrinsics.
CHANGE_EXTRINSICS = False

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
    return bboxes.reshape((-1, 4))

def load_extrinsics():
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

def load_point_cloud():
    points = []
    point_cloud_colors = []
    with open("Corpus_3D.txt", 'r') as f:
        count = 0
        for l in f:
            if count > 0:
                x, y, z, r, g, b= l.strip().split(" ")
                point_cloud_colors.append((int(r)/255, int(g)/255, int(b)/255))
                points.append([float(x), float(y), float(z)])
            count += 1

    points = np.array(points)
    points *= SFM_SCALE
    return points, point_cloud_colors

def visualize_point_cloud(points, colors, transform_m):
    plt.figure()
    ax = plt.axes(projection='3d')
    print(points.shape)

    points_h = np.hstack((points, np.ones(len(points)).reshape(-1,1)))
    points_trans = points_h @ transform_m.T
    points_trans = points_trans[:, :3] / points_trans[:, 3].reshape(-1,1)
    ax.scatter(points_trans[:,0], points_trans[:,1], points_trans[:,2], s=0.1,c=colors)

    # for SFM_SCALE=1720
    # ax.set_xlim(-30000,5000)
    # ax.set_ylim(-20000,5000)
    # ax.set_zlim(-5000,40000)

    # for SFM_SCALE=1
    ax.set_xlim(-4000,200)
    ax.set_ylim(-100,400)
    ax.set_zlim(-10,3000)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

K, r0, r1, r2, t0, t1 = get_intrinsic_m()
dist_coeff = np.array([r0, r1, t0, t1, r2])

extrinsics = load_extrinsics()    
ex_m_first = get_extrinsic_matrix(*extrinsics[0])
ex_m_first = np.vstack((ex_m_first, [[0,0,0,1]]))
ex_m_first_inv = np.linalg.inv(ex_m_first)
print(ex_m_first @ ex_m_first_inv)
print('******')

point_cloud, point_cloud_colors = load_point_cloud()

track_file_name = 'lag_5_cross_check_0.3_track_results_merged_algo.json' 
with open(track_file_name, 'r') as f:
    track_results = json.load(f)
img_names = sorted(track_results.keys())
img_names = img_names[:30]

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

tracklets_dict = get_tracklets()
tracklet_ids = set(tracklets_dict.keys())

num_colors = 200
colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(num_colors)]

# get transformation for undistortion
img_size = (1920, 1080)
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, img_size, alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, np.eye(3), new_K, img_size, cv2.CV_32FC1)
print("original K")
print(K)
print("new K")
print(new_K)

def get_tracklet_traj(id, smooth_weight):
    bboxes = [x['bbox'] for x in tracklets_dict[id]]
    fids = [x['fid'] for x in tracklets_dict[id]]

    bboxes = smooth_tracklet(bboxes, sigma=1)

    # undistort bboxes
    bboxes = undistort_tracklet(bboxes, K, dist_coeff, new_K)

    if DEBUG:
        # if os.path.exists('./debug'):
            # shutil.rmtree('./debug')
        # os.mkdir('debug')

        if not os.path.exists('./debug'):
            os.mkdir('debug')

        for idx in range(len(fids)):
            fid = fids[idx]
            bbox = bboxes[idx]
            bbox = [int(x) for x in bbox]

            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            long = max(h,w)

            img_name = img_names[fid]
            img = cv2.imread('data/ny_china_town_0350_0420/images/' + img_name)

            img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            # img = cv2.undistort(img, K, dist_coeff)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colors[id%num_colors], thickness=2)
            cv2.putText(img, str(id), (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=long/200+0.5, color=colors[id%num_colors], thickness=2)
            cv2.imwrite('debug/' + img_name, img)

    # get pseudo 3D center points
    Ps = []
    bboxes_w = []
    bboxes_h = []
    for idx in range(len(fids)):
        fid = fids[idx]
        if fid == 0: # numerical issue for 0th frame
            continue

        extrinsic = extrinsics[fid]
        bbox = bboxes[idx]
        c_x = (bbox[0] + bbox[2])/2
        c_y = (bbox[1] + bbox[3])/2
        c_h = np.array([c_x, c_y, 1]).reshape(-1, 1) 
        bboxes_w.append(bbox[2] - bbox[0])
        bboxes_h.append(bbox[3] - bbox[1])

        ex_m = get_extrinsic_matrix(*extrinsic)

        if CHANGE_EXTRINSICS:
            ex_m = np.vstack((ex_m, [[0,0,0,1]]))
            ex_m = ex_m @ ex_m_first_inv 
            ex_m = ex_m[:3]

            M = new_K @ ex_m
            M_inv = np.linalg.pinv(M)
            P_h = M_inv @ c_h 
        else:
            M = new_K @ ex_m
            M_inv = np.linalg.pinv(M)
            P_h = M_inv @ c_h 

            # make lifted points be relative to the first camera origin
            P_h = ex_m_first @ P_h # (4, 1)

        P_h /= P_h[3]
        Ps.append(P_h[:3].flatten())
    Ps = np.array(Ps)

    # optimize
    scales = cp.Variable(len(Ps))
    bboxes_w = np.array(bboxes_w)
    bboxes_h = np.array(bboxes_h)

    smooth_term = 0
    for i in range(1, len(Ps)):
        smooth_term += cp.sum((scales[i] * Ps[i] - scales[i-1]*Ps[i-1])**2)

    # metric_term_w = cp.multiply(scales, Ps[:, 2].reshape(-1))/HEAD_W - K[0][0] / bboxes_w
    # metric_term_h = cp.multiply(scales, Ps[:, 2].reshape(-1))/HEAD_H - K[1][1] / bboxes_h

    f_avg = (new_K[0][0] + new_K[1][1]) / 2
    metric_term_w = cp.multiply(scales, Ps[:, 2].reshape(-1))/HEAD_W - f_avg / bboxes_w
    metric_term_h = cp.multiply(scales, Ps[:, 2].reshape(-1))/HEAD_H - f_avg / bboxes_h

    objective = cp.Minimize(smooth_weight*smooth_term + cp.sum_squares(metric_term_w) + cp.sum_squares(metric_term_h))
    constraints = []
    prob = cp.Problem(objective, constraints)
    print("Optimal value", prob.solve())
    print("Optimal var")
    print(scales.value)

    real_Ps = []
    for i in range(len(Ps)):
        real_Ps.append(Ps[i]*scales.value[i])

    real_Ps = np.array(real_Ps)
    return real_Ps


def plot_tracklet_trajs(ids, colors, smooth_weight, elev, azim, filename, title):
    plt.figure()
    # ax = visualize_point_cloud(point_cloud, point_cloud_colors, ex_m_first)
    ax = plt.axes(projection='3d')
    for id, c in zip(ids, colors):
        real_Ps = get_tracklet_traj(id, smooth_weight)
        for i in range(0, len(real_Ps)):
            if i == 0:
                ax.plot(real_Ps[i:i+2, 0].flatten(), real_Ps[i:i+2, 1].flatten(), real_Ps[i:i+2, 2].flatten(), 'go-')
            else:
                ax.plot(real_Ps[i:i+2, 0].flatten(), real_Ps[i:i+2, 1].flatten(), real_Ps[i:i+2, 2].flatten(), c+'o-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    ax.view_init(elev=elev, azim=azim,vertical_axis='y')
    ax.invert_zaxis()
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()



smooth_weight = 0.001
# bird eye view
elev = 90
azim = 0
# 3d view
## elev = 30
## azim = 10
out_dir = 'tracklet_traj_plots_sfm_{}_change_e_{}'.format(SFM_SCALE, str(CHANGE_EXTRINSICS))
# out_dir = 'tracklet_traj_plots_sfm_{}_change_e_{}_pc'.format(SFM_SCALE, str(CHANGE_EXTRINSICS))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# plot_tracklet_trajs([1], ['r'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(1, smooth_weight, elev, azim)), 'id: {}'.format(1))
# plot_tracklet_trajs([322], ['r'], smooth_weight, elev, azim, os.path.join(out_dir,"traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(322, smooth_weight, elev, azim)), 'id: {}'.format(322))
# plot_tracklet_trajs([491], ['r'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(491, smooth_weight, elev, azim)), 'id: {}'.format(491))

# plot_tracklet_trajs([2], ['b'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(2, smooth_weight, elev, azim)), 'id: {}'.format(2))
# plot_tracklet_trajs([492], ['b'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(492, smooth_weight, elev, azim)), 'id: {}'.format(492))

# plot_tracklet_trajs([4], ['y'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(4, smooth_weight, elev, azim)), 'id: {}'.format(4))
# plot_tracklet_trajs([324], ['y'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(324, smooth_weight, elev, azim)), 'id: {}'.format(324))
# plot_tracklet_trajs([428], ['y'], smooth_weight, elev, azim, os.path.join(out_dir, "traj_id_{}_smooth_{}_elev_{}_azim_{}.jpg".format(428, smooth_weight, elev, azim)), 'id: {}'.format(428))
# # plot_tracklet_trajs([1, 322, 491], ['r']*3, smooth_weight, elev, azim, os.path.join(out_dir, "traj_all_smooth_{}_elev_{}_azim_{}.jpg".format(smooth_weight, elev, azim)), 'all traj')

# all_persons = [1, 322, 491]
# all_colors = ['r']*3

all_persons = [1, 322, 491]+[2, 492]+[4, 324, 428] + [69, 624] + [50, 123, 183, 252, 301, 345, 399]
all_colors = ['r']*3+['b']*2+['y']*3 + ['c'] * 2 + ['m'] * 7
plot_tracklet_trajs(all_persons, all_colors, smooth_weight, elev, azim, os.path.join(out_dir, "traj_all_smooth_{}_elev_{}_azim_{}.jpg".format(smooth_weight, elev, azim)), 'all traj')

