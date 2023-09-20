import pandas as pd
import shutil
import os
import re
from glob import glob
from sklearn.model_selection import train_test_split

img_dir = './data/blue_images'
train_dir = './data/train'
val_dir = './data/val'
test_dir = './data/test'

filenames = [i.split('/')[-1] for i in glob(img_dir+'/*')]

# Read in data
# df = pd.read_csv('data/power_analysis/img_data.csv')
df = pd.read_csv('./data/power_analysis/img_data.csv')

# Set train/validation 80:20 splits, stratified over experimental factors
train, val = train_test_split(df, test_size=.2, train_size=.8, random_state=345, stratify=df[['variety', 'method', 'sample_mass_g']])

# Create a test group of 20 random images over variety and method
_, test = train_test_split(df, test_size=0.04, random_state=345, stratify=df[['variety', 'method']])


if __name__ == '__main__':
    results = []
    # Move training images
    train_ids = list(train['img_id'].unique())

    for img_id in train_ids:
        r = re.compile(f"{img_id}.*")
        newlist = list(filter(r.match, filenames)) # Read Note below
        img_name = newlist[0]
        img_row = {'img_id': img_id, 'img_name': img_name, 'class': 'train'}
        results.append(img_row)
        shutil.copy2(os.path.join(img_dir, img_name), os.path.join(train_dir, img_name))


    # Move validation images
    val_ids = list(val['img_id'].unique())

    for img_id in val_ids:
        r = re.compile(f"{img_id}.*")
        newlist = list(filter(r.match, filenames)) # Read Note below
        img_name = newlist[0]
        img_row = {'img_id': img_id, 'img_name': img_name, 'class': 'val'}
        results.append(img_row)
        shutil.copy2(os.path.join(img_dir, img_name), os.path.join(val_dir, img_name))


    # Move random images for inference
    test_ids = list(test['img_id'].unique())

    for img_id in test_ids:
        r = re.compile(f"{img_id}.*")
        newlist = list(filter(r.match, filenames)) # Read Note below
        img_name = newlist[0]
        shutil.copy2(os.path.join(img_dir, img_name), os.path.join(test_dir, img_name))
    
    results = pd.DataFrame.from_dict(results)
    results = results.merge(df, left_on='img_id', right_on='img_id')
    results.to_csv('./data/power_analysis/results.csv')