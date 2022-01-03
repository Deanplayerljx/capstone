# OCTrack: 3D Human Tracking under Severe Occlusion
The code is based on [vedadet](https://github.com/Media-Smart/vedadet).

### Environment Set Up
a.Create a conda virtual environment and activate it.
```
conda create -n octrack python=3.8.5 -y
conda activate octrack
```
b.Install PyTorch (1.6.0 or higher) and torchvision. Example commands for various versions.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
```
c. Clone the repository.
```
git clone https://github.com/Deanplayerljx/capstone.git
cd capstone
```
d. Install vedadet (already part of the repository).
```
pip install -r requirements/build.txt
pip install -v -e .
```
e. Install additional dependencies.
```
pip install tqdm
pip install opencv-contrib-python
pip install cvxpy
pip install --upgrade numpy
```

### Data
We used a short clip from [this video](https://www.youtube.com/watch?v=VD6Fc5d1VFU) (3:50-4:20) for testing. The clip can be found [here](https://drive.google.com/file/d/1Gv4O1XOem-Jwp0ICDQegC6k23MgvIUpC/view?usp=sharing). We downsampled the clip to 30 fps by saving every other frame. The extracted images can be found [here](https://drive.google.com/file/d/1xEPNLovIiK4r8-PxiaDqeap6iY4_4gdZ/view?usp=sharing)

Dataset for detection is located in [here](https://drive.google.com/file/d/1C550hilabCcMl56DZ58enhJw_usoUKJY/view?usp=sharing). Please download and extract to `data/face_clip` before starting to do detection. 

### Detection
Run the following command to traing tinaface detection on go-pro captured head dataset
```
python tools/trainval.py configs/trainval/tinaface/tinaface_r50_fpn_gn_dcn_custom.py --workdir YOUR_MODEL_DIR
```

Run the following command to evaluate the result
```
python tools/test.py configs/trainval/tinaface/tinaface_r50_fpn_gn_dcn_custom.py MODEL_CHECKPOINT IOU_THRESHOLD --out OUTPUT_RESULT_DICT
```

Run the following command for inference images, the model checkpoint is specified in config file

```
python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py INPUT_IMG_DIR OUT_IMG_DIR
```

<img src=demo/detection_results.jpg width="50%">

Please download weights [here](https://drive.google.com/file/d/1hFSGSTHTYxdoJ7qAlkbPWOYtJDA4IRJG/view?usp=sharing). 

### Tracking
Run the following command to get fixed-lag tracking results:
```
python fixed_lay_track.py
```
Run the following command to merge overlapping tracklets:
```
python merge_overlap_2d_tracks_algo.py
```
To visulize the tracking results, run the following command:
```
python visualize_tracking.py
```
Run the following command to lift tracklets to 3D (preliminary):
```
python lift_3d.py
```