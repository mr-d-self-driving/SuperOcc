# Prepare Dataset

## Nuscenes
1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. (Optional) Download Occ3d-nuScenes from [link](https://tsinghua-mars-lab.github.io/Occ3D/) and place it in `data/nuscenes/gts`
3. (Optional) Download SurroundOcc from [link](https://github.com/weiyithu/SurroundOcc) and place it in `data/nuscenes/surround_occ`.
4. Prepare data with scripts provided by mmdet3d:
```
mim run mmdet3d create_data nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
5. Perform data preparation for SupeOcc:
```
python tools/create_data_nusc.py
```
6. Folder structure:
```
data/nuscenes
├── maps
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── samples
├── sweeps
├── gts
├── surround_occ
├── v1.0-test
└── v1.0-trainval
