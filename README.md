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
# Usage
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
