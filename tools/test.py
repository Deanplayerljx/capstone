import argparse

import torch
import pickle

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("iou", default=0.5, type=float, help="Iou Threshold for AP calculation")
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--precal-det', help="pre-calculate detection same as output format")

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader



def test(engine, data_loader):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = engine(data)[0]
        results.append(result)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint)
    if args.precal_det is not None:
        with open(args.precal_det, "rb") as f:
            results = pickle.load(f)
    else:
        results = test(engine, data_loader)
    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)
    iou_thr = args.iou
    print("AP@Small")
    data_loader.dataset.evaluate(results, bboxes_size_range=(0, 765), iou_thr=iou_thr)
    print("AP@Medium")
    data_loader.dataset.evaluate(results, bboxes_size_range=(765, 2114), iou_thr=iou_thr)
    print("AP@Large")
    data_loader.dataset.evaluate(results, bboxes_size_range=(2114, 138508), iou_thr=iou_thr)
    print("AP")
    data_loader.dataset.evaluate(results, bboxes_size_range=(0, 138508), iou_thr=iou_thr)

if __name__ == '__main__':
    main()
