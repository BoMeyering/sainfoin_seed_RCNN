import pandas as pd
import numpy as np
from glob import glob

img_dir = '../data/blue_images/'

annot_path = '../data/annotations/annotations_export.csv'

filenames = [i.split('/')[-1] for i in glob(img_dir+'*')]

# print(filenames)

annotated = pd.read_csv(annot_path)

a_filenames = list(annotated.img_id.unique())

# print(a_filenames)

filenames = set(filenames)
a_filenames = set(a_filenames)

diff = filenames.difference(a_filenames)
print(diff)

diff2 = a_filenames.difference(filenames)
print(diff2)
