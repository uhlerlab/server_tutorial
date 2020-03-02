# Tutorial for using the GPU Server


## Installing PyTorch
(1) Install anaconda. To do this, first run the following command on your server:

    `wget https://repo.anaconda.com/archive/Anaconda3-2019.10-MacOSX-x86_64.sh`

Then follow the instructions at https://docs.anaconda.com/anaconda/install/linux/ for python3.7 .
  
(2) After installing conda, create a conda environment (put in a reasonable name for myenv) with:

`conda create -n myenv python=3.7`

(3) Check the cuda version - you can just check /usr/local/ for the cuda version. 

(4) Go to the PyTorch download page: https://pytorch.org/get-started/locally/ and download the correct version.  For uhlergroup.mit.edu this should be:

`conda install pytorch torchvision cudatoolkit=9.2 -c pytorch`

For the new server this should be:

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

(5) To test the installation, just run: `python` followed by `import torch`


