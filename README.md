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

# 
