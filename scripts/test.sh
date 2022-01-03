cd ..

### evaluation
# python tools/test.py configs/trainval/tinaface/tinaface_r50_fpn_gn_dcn_custom.py model/y.pth 0.5 --out output.pkl


### visualization
python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py /home/rawal/Desktop/capstone/data/chinatown_images output