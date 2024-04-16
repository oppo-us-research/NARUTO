ROOT=${PWD} 

### create conda environment ###
conda create -y -n naruto python=3.8 cmake=3.14.0

### activate conda environment ###
source activate naruto

# ### Setup habitat-sim ###
cd ${ROOT}/third_parties
git clone git@github.com:Huangying-Zhan/habitat-sim.git habitat_sim
cd habitat_sim
pip install -r requirements.txt
python setup.py install --headless --bullet

### extra installation ###
pip install opencv-python
conda install -y ipython
pip install mmcv==2.0.0

### CoSLAM installation ###
cd ${ROOT}/third_parties/coslam
git checkout 3bb904e
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
cd external/NumpyMarchingCubes
python setup.py install

### NARUTO installation ###
pip install -r ${ROOT}/envs/requirements.txt
