# LG-SGNet

This repo is the official implementation for <mark>LG-SGNet: LG-SGNet: Local and Global Self-Attention Enhanced Graph Convolutional Network for Skeleton-based Action Recognition</mark>. 

# Architecture of LG-SGNet
![image](https://github.com/DingYyue/LG-SGNet/src/framework.png)
![image](https://github.com/DingYyue/LG-SGNet/src/LG-GCN.png)
![image](https://github.com/DingYyue/LG-SGNet/src/LG-TCN-branch1.png)

# Data Preparation
### There are 3 datasets to download:
- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

### Data Processing
##### Directory Structure
- Put downloaded data into the following directory structure:
  ```
  - data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
  
##### Generating Data
- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
  ```
   cd ./data/ntu # or cd ./data/ntu120
  # Get skeleton of each performer
  python get_raw_skes_data.py
  # Remove the bad skeleton 
  python get_raw_denoised_data.py
  # Transform the skeleton to the center of the first frame
  python seq_transformation.py
  
# Pretrained Model
[pretrained_model](https://pan.baidu.com/s/1GmmMIJKPCDXnnCSCWEWIPg?pwd=5136)
# Training & Testing
### Training
- Example: training LG-SGNet on NTU RGB+D 120 cross subject
  ```
  python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir ./work_dir/ntu120/joint --model model.LG-SGNet.Model --weights pretrained_model/...
  
### Testing:
- Example: testing LG-SGNet on NTU RGB+D 120 cross subject
  ```
  python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
  
- Ensemble the results of different modalities
  ```
  python ensemble.py --dataset ntu120/xsub --joint-dir work_dir/ntu120/joint --bone-dir work_dir/ntu120/bone --joint-motion-dir work_dir/ntu120/motion --bone-motion-dir work_dir/ntu120/bone_motion

# Acknowledge
This repo is based on CTR-GCN, thanks to their excellent work.
