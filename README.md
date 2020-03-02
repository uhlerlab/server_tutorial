# Tutorial for using the GPU Server


## Installing PyTorch
1. Install anaconda. To do this, first run the following command on your server:

    `wget https://repo.anaconda.com/archive/Anaconda3-2019.10-MacOSX-x86_64.sh`

Then follow the instructions at https://docs.anaconda.com/anaconda/install/linux/ for python3.7 .
  
2. After installing conda, create a conda environment (put in a reasonable name for myenv) with:

`conda create -n myenv python=3.7`

3. Check the cuda version - you can just check /usr/local/ for the cuda version. 

4. Go to the PyTorch download page: https://pytorch.org/get-started/locally/ and download the correct version.  For uhlergroup.mit.edu this should be:

`conda install pytorch torchvision cudatoolkit=9.2 -c pytorch`

For the new server this should be:

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

5. To test the installation, just run: `python` followed by `import torch`

## Useful Tools/Commands
1. `screen` - use this to run jobs in the background.  You can also use `nohup` if you like, but I prefer screen.
*`screen -r screen_number` lets you select a screen to restart. If you don't remember the screen number, just type `screen -list` or `screen -r` to output a list of screens available. 
* Use cntr-a-d to exit a screen.   
* One screen is typically up to keep visdom running, but I will discuss this later. 
2. `nvidia-smi` - use this to check GPU memory usage, utilization, temperature, etc.  
3. `export CUDA_VISIBLE_DEVICES=gpu_num` - use this to select a specific GPU to run your job on (replace gpu_num with a whole number - i.e. 0, 1, 2, ...).  
4. `python -u fname.py` - try to use the "-u" flag when running python so that outputs are flushed to the screen.
5. `python -u fname.py | tee log_dir/log_file.out` - the tee command composed with a pipe lets you write to a file while simulatenously printing to the screen.  Please try to log all experiments and store log files in a directory so that it is easy to clean up if needed.  


## Training an Autoencoder on 1 Image
