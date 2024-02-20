# sainfoin_seed_RCNN
Code for training a Faster RCNN to detect and classify sainfoin seeds in experimental images.

# Repo Cloning
Clone this repo to a desired location in your local environment.
```
~/$ git clone git@github.com:BoMeyering/sainfoin_seed_RCNN.git
```

# Data Acquisition
The entire dataset with all annotations are available in Zenodo at [Here](https://doi.org/10.5281/zenodo.8346923)
Or just grab both files in your terminal. `cd` into the data folder in the repo directory you just cloned.
```
~/$ cd sainfoin_seed_RCNN/data
~/sainfoin_seed_RCNN/data$ wget https://zenodo.org/record/8346924/files/train_val_images.zip https://zenodo.org/record/8346924/files/seed_weights.csv
```

The zip file is about 1Gb so it might take some time to download depending on your connection speed.

Extract all of the images and :
```
~/sainfoin_seed_RCNN/data$ unzip train_val_images.zip
```

# Create a new environment
Lots of ways to do this, choose your favorite one (`virtualenv`, `venv`, `conda`, etc.). I like managing python installations with `pyenv` as it is easy to use and set up directory specific environments. Check it out [here](https://realpython.com/intro-to-pyenv/). I'll use pyenv to create a Python 3.11.5 virtualenv called `pytorch_env`.
```
~/sainfoin_seed_RCNN$ pyenv install 3.11.5
~/sainfoin_seed_RCNN$ pyenv virtualenv 3.11.5 pytorch_env
~/sainfoin_seed_RCNN$ pyenv local pytorch_env
```
And now the virtualenv is activated. pip install the dependencies
```
(pytorch_env) ~/sainfoin_seed_RCNN$ pip install requirements.txt
``` 

This directory is now set up to run all python scripts within the directory in the `pytorch_env` environment, hence the `(pytorch_env)` prepended to the terminal line. You can deactivate it any time by calling ```source deactivate``` and activate it again using the ```pyenv local ...``` command above.

# Image Inference

# Environment Setup
Clone this repo to your local environment
```
~/$ git clone https://github.com/TLILegume/sainfoin_seed_RCNN.git
~/$ cd sainfoin_seed_RCNN
```

I would suggest setting up a virtualenv on your machine using pyenv, venv, or virtualenv, whatever you prefer. 
Using pyenv with a Python 3.11.6 installation, you can set up a virtualenv named 'sainfoin_RCNN' shown below
```
~/sainfoin_seed_RCNN$ pyenv virtualenv 3.11.6 sainfoin_RCNN
~/sainfoin_seed_RCNN$ pyenv local sainfoin_RCNN
(sainfoin_RCNN) ~/sainfoin_seed_RCNN$
```
Then pip install all the requirements for the project


# Get Data and Model Weights
## On UNIX systems
Run the bash script at ```./download_data_weights.sh```
This script checks and downloads all of the appropriate files, images, and model weights into the main folder of the repo.

## On Windows
You can either run the Bash script in WSL if you have it installed, or you can download the files directly and extract them in the appropriate directories.
You can download all the images from [Zenodo][def]. Create a folder ```./sainfoin_seed_RCNN/data/images``` and extract them there.
The model checkpoint can be downloaded [here][def2]. You can choose to download one or several of the checkpoints. Create the directory ```./sainfoin_seed_RCNN/model_chkpt``` and extract the .pth files directly in the folder like this ```./model_chkpt/frcnn_sainfoin_1.0_100.pth```.

# Set

[def]: https://zenodo.org/doi/10.5281/zenodo.8346923
[def2]: https://zenodo.org/doi/10.5281/zenodo.8387982