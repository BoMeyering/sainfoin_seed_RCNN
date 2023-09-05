import ndjson
import shutil
import pandas as pd
from glob import glob
import os

annotations_path = '../data/annotations/annotations_export.csv'
img_dir = '../data/all_images'
move_dir = '../data/annotated_images'

file_paths = glob(os.path.join(img_dir, "*"))
file_names = [i.split('/')[-1] for i in file_paths]
print(file_names)

annot_df = pd.read_csv(annotations_path)

print(annot_df)

img_ids = list(annot_df['img_id'].unique())
print(img_ids)

for img_id in img_ids:
	# print(img_id)
	if img_id in file_names:
		print(True)
	shutil.copy2(os.path.join(img_dir, img_id), os.path.join(move_dir, img_id))


