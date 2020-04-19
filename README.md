# CatNet
Implementation of our paper CatNet: Class Incremental 3D ConvNets for Lifelong Egocentric Gesture Recognition
## Requirements
In the CatNet.Dockerfile.
Python3\
torch==1.2\
torchvision==0.4.0\
scipy\
pillow==6.2.1\
sklearn\
tqdm\
torchsummary\
matplotlib\
opencv-python-headless\
pandas\
scikit-image
## Usage
* Create annotation files for EgoGesture
  ```
  Change the frame_path and label_path to your own path in the create_annotation.py
  python3 create_annotation.py
  EgoGesture dataset folder structure
  |
  |-frames
  |–--Subject01
  |--- ......
  |-labels
  |---Subject01
  |--- .....
  ```

* Train task0 model
```
For ResNext-101-32
python3 train_R3D_task0.py --is_train True --n_frames_per_clip 32 --pretrain_path models/pretrained_models/jester_resnext_101_RGB_32.pth --modality Depth (RGB or RGB-D)

For ResNext-101-16
python3 train_R3D_task0.py --is_train True --n_frames_per_clip 16 --pretrain_path models/pretrained_models/resnext-101-kinetics.pth --modality Depth (RGB or RGB-D)

For ResNet-50-16
python3 train_R3D_task0.py --is_train True --n_frames_per_clip 16 --pretrain_path models/pretrained_models/resnet-50-kinetics.pth --arch resnet-50 --model resnet --model_depth 50 --modality Depth (RGB or RGB-D)
```


## Acknowledgement
We thank Kensho Hara for releasing [3D-ResNets-PyTorch Repo](https://github.com/kenshohara/3D-ResNets-PyTorch) and Okan Köpüklü for releasing [Real-time-GesRec Repo](https://github.com/ahmetgunduz/Real-time-GesRec), which we build our work based on their work. 
