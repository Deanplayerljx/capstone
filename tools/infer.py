import argparse
import cv2
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import pickle

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('imgname', help='image file name')
    parser.add_argument('img_folder', help='image file name')
    parser.add_argument("out_base", help="out file name")
    parser.add_argument("--precal-det", help="pre calculate prediction")
    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    # load_weights(engine.model, cfg.weights.filepath)

    model_filepath = 'model/y.pth'
    load_weights(engine.model, model_filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 2

    
    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        # label_text = class_names[
        #     label] if class_names is not None else f'cls {label}'
        label_text = '' 
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)

    
    
def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # imgname = args.imgname
    img_folder = args.img_folder
    img_base_dir = '/'.join(img_folder.split('/')[:-1])
    
    class_names = cfg.class_names

    test_cfg = cfg.infer_engine.test_cfg
    print(test_cfg)

    engine, data_pipeline, device = prepare(cfg)

    write_image = True
    result_dict = {}
    # print(sorted(os.listdir(img_folder)))
    out_folder = "tina_face_finetune_{}_{}".format(test_cfg.score_thr, test_cfg.nms.iou_thr)
    out_folder = os.path.join(args.out_base, out_folder)
    if not os.path.exists(out_folder):
        # os.mkdir(out_folder)
        os.makedirs(out_folder)
    if args.precal_det is not None:
        with open(args.precal_det, "rb") as f:
            results = pickle.load(f)
    for idx, file in enumerate(tqdm(sorted(os.listdir(img_folder)))):

        if file.endswith('.jpg'):
            result_dict[file] = []
            img_path = os.path.join(img_folder, file)
            data = dict(img_info=dict(filename=img_path), img_prefix=None)

            data = data_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if device != 'cpu':
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                # just get the actual data from DataContainer
                data['img_metas'] = data['img_metas'][0].data
                data['img'] = data['img'][0].data
            if args.precal_det is not None:
                result = results[file]
            else:
                result = engine.infer(data['img'], data['img_metas'])[0]
            # print(result)
            if write_image:
                plot_result(result, img_path, class_names, outfp=os.path.join(out_folder, file))

            out_img_name = file
            bboxes = np.vstack(result)
            for bbox in bboxes:
                bbox_int = bbox[:4].astype(np.int32)
                result_dict[file].append(bbox_int.tolist() + [float(bbox[4])])
            if idx % 1000 == 0:
                with open(os.path.join(out_folder, "result_{}.json".format(idx)), 'w') as f:
                    json.dump(result_dict, f)

    with open(os.path.join(out_folder, "result_full.json"), 'w') as f:
        json.dump(result_dict, f)



if __name__ == '__main__':
    main()