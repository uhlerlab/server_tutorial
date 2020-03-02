# Tutorial for using the GPU Server
Adit: these are my suggestions for using the server effectively, but send me a message if there are other useful tips you feel are important.  

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


## Generally Useful Tools/Commands
1. `screen` - use this to run jobs in the background.  You can also use `nohup` if you like, but I prefer screen.
    * `screen -r screen_number` lets you select a screen to restart. If you don't remember the screen number, just type `screen -list` or `screen -r` to output a list of screens available. 
    * Use cntr-a-d to exit a screen.   
    * Example: `python -m visdom.server` run once through 1 screen keeps a visdom server up.  
2. `python -u fname.py` - try to use the "-u" flag when running python so that outputs are flushed to the screen.
3. `python -u fname.py | tee log_dir/log_file.out` - the tee command composed with a pipe lets you write to a file while simulatenously printing to the screen.  
4. `htop` - this allows you to monitor CPU core and memory usage.  You can also see other jobs.  That being said, please do not kill anyone else's jobs without consulting them first.  


## GPU Specific Tools/Commands
1. `nvidia-smi` - use this to check GPU memory usage, utilization, temperature, etc.  
2. `export CUDA_VISIBLE_DEVICES=gpu_num` - use this to select a specific GPU to run your job on (replace gpu_num with a whole number - i.e. 0, 1, 2, ...).  
3. `torch.backends.cudnn.enabled = True` - set this in your main.py file to make sure that you use cudnn for training.  
4. `torch.backends.cudnn.benchmark = True` - set this in your main.py if the inputs/outputs of your neural network are unchanging. 
    * Note the first epoch may be slower, but 2nd epoch onwards will be faster.  
    * In my experience, this is usually not worth using unless your job is going to take more than a few minutes.

## Editor suggestions
I think most people in the group use Emacs or Vim, but let me know if there are other suggestions.

1. As I am an Emacs user, I've gone ahead and installed Emacs prelude on the new server.  I'll try to do this on the old server when I get a chance.  Emacs also lets you edit remotely and push changes to the server upon saving.  This helps somewhat if you have a bad connection.  

## Good Practices 
These should really be required, but are often forgotten when running quick experiments.  
1.  Please use random seeds to make sure all experiments are reproducable.  Here is a list of seeds to set:
    * torch.manual_seed(seed)
    * np.random.seed(seed)
    * torch.cuda.manual_seed(seed)
2.  Please try to log all experiments and store log files in a directory so that it is easy to clean up if needed.  
3.  Do **not** place all your code in 1 main file.  This is terrible for anyone else to disentangle.  Instead, I propose the following format, but let me know if you have better suggestions:
    1. main.py - this should be very short; set your seeds, load your data, and run the model trainer
    2. dataset.py - process your dataset and return train/test splits
    3. model.py - code for your network architecture/model
    4. trainer.py - code to train your neural network
4.  Checkpoint (save) your models.  This is invaluable if the server goes down.  **Important** - If you need to checkpoint every epoch, please clean up the unused checkpoints to free up space on the server.  
