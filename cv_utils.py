import numpy as np
import torch

def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if boxAArea < 0.1 or boxBArea < 0.1:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def tracklet_overlap_iou(tracklet1, tracklet2):
    """
    tracklet1 and tracklet2 should end at the same timestamp
    """
    # get overlapping sequence
    min_len = min(len(tracklet1), len(tracklet2))
    if min_len == 0:
        return 0
    mean_iou = sequence_mean_iou(tracklet1[len(tracklet1)-min_len:],
                tracklet2[len(tracklet2)-min_len:])
    return mean_iou
    

def sequence_mean_iou(s1, s2):
    assert(len(s1) == len(s2))
    total_iou = 0
    for bbox1, bbox2 in zip(s1, s2):
        total_iou += iou(bbox1, bbox2)
    return total_iou / len(s1)

def nms_pytorch(P,scores, thresh_iou):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """

    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    # scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
    

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(P[idx])

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w*h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        
        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
    
    return keep


def tracklet_nms(tracklets, scores, thresh_iou):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """

    # we extract coordinates for every 
    # prediction box present in P
    # x1 = tracklets[:, 0]
    # y1 = tracklets[:, 1]
    # x2 = tracklets[:, 2]
    # y2 = tracklets[:, 3]

    # we extract the confidence scores as well
    # scores = P[:, 4]

    # calculate area of every block in P
    # areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    # order = scores.argsort()
    order = np.argsort(scores)

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
    

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        # keep.append(P[idx])
        keep.append(idx)

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        
        overlap_iou = []
        remain_tracklets = tracklets[order]
        for remain_tracklet in remain_tracklets:
            mean_iou = tracklet_overlap_iou(remain_tracklet, tracklets[idx])
            overlap_iou.append(mean_iou)
        overlap_iou = np.array(overlap_iou)
        
        # select coordinates of BBoxes according to 
        # the indices in order
        # xx1 = torch.index_select(x1,dim = 0, index = order)
        # xx2 = torch.index_select(x2,dim = 0, index = order)
        # yy1 = torch.index_select(y1,dim = 0, index = order)
        # yy2 = torch.index_select(y2,dim = 0, index = order)

        # # find the coordinates of the intersection boxes
        # xx1 = torch.max(xx1, x1[idx])
        # yy1 = torch.max(yy1, y1[idx])
        # xx2 = torch.min(xx2, x2[idx])
        # yy2 = torch.min(yy2, y2[idx])

        # # find height and width of the intersection boxes
        # w = xx2 - xx1
        # h = yy2 - yy1
        
        # # take max with 0.0 to avoid negative w and h
        # # due to non-overlapping boxes
        # w = torch.clamp(w, min=0.0)
        # h = torch.clamp(h, min=0.0)

        # # find the intersection area
        # inter = w*h

        # # find the areas of BBoxes according the indices in order
        # rem_areas = torch.index_select(areas, dim = 0, index = order) 

        # # find the union of every prediction T in P
        # # with the prediction S
        # # Note that areas[idx] represents area of S
        # union = (rem_areas - inter) + areas[idx]
        
        # # find the IoU of every prediction in P with S
        # IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        # mask = IoU < thresh_iou
        mask = overlap_iou < thresh_iou
        order = order[mask]
    
    return keep