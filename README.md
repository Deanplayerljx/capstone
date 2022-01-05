# capstone
The code is based on [vedadet](https://github.com/Media-Smart/vedadet).

### Environment Set Up
a.Create a conda virtual environment and activate it.
```
conda create -n vedadet python=3.8.5 -y
conda activate vedadet
```
b.Install PyTorch (1.6.0 or higher) and torchvision.
c. Clone the repository.
```
git clone https://github.com/Deanplayerljx/capstone.git
cd capstone
```
d. Install vedadet.
```
pip install -r requirements/build.txt
pip install -v -e .
```
e. Install additional dependencies.
```
pip install tqdm
pip install opencv-contrib-python
pip install cvxpy
```

### Data
We used a short clip from [this video](https://www.youtube.com/watch?v=VD6Fc5d1VFU) (3:50-4:20) for testing. The clip can be found [here](https://drive.google.com/file/d/1Gv4O1XOem-Jwp0ICDQegC6k23MgvIUpC/view?usp=sharing). We downsampled the clip to 30 fps by saving every other frame. The extracted images can be found [here](https://drive.google.com/file/d/1xEPNLovIiK4r8-PxiaDqeap6iY4_4gdZ/view?usp=sharing). Please download the images and put them under the ```data``` folder.

### Tracking
Run the following command to get fixed-lag tracking results:
```
python fixed_lay_track.py --det_file det/file/path --img_dir img/dir/path --out_file output/file/path
```
Run the following command to merge overlapping tracklets:
```
python merge_overlap_2d_tracks_algo.py --track_file fixed/lag/track/file/path --out_file output/file/path
```
To visulize the tracking results, run the following command:
```
python visualize_tracking.py --track_file merged/track/file/path --img_dir img/dir/path --out_dir output/folder
```
Run the following command to lift tracklets to 3D (preliminary):
```
python lift_3d.py --track_file merged/track/file/path --out_dir output/folder
```