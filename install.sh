# flwr, torch, torchvision, mmcv, mmaction2, 
# opencv, numpy, pandas, yaml 
pip install numpy pandas sklearn
pip install opencv-python
pip install PyYAML

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install Pillow==7.0.0

pip3 install openmim
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .

pip install flwr