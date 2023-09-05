import labelbox as lb
from _labelbox_config import LB_API_KEY, PROJECT_ID, DATASET_ID
import json
import os
import pandas as pd

import ndjson

annotation_dir = '../data/annotations/'

# ndjson_output = os.path.join(annotation_dir, 'annotations_export.ndjson')
ndjson_output = os.path.join(annotation_dir, 'export-result.ndjson')

# client = lb.Client(api_key = LB_API_KEY)
# project = client.get_project(PROJECT_ID)
# labels = project.export_v2(params={
# 	"data_row_details": True,
# 	"metadata_fields": True,
# 	"attachments": True,
# 	"project_details": True,
# 	"performance_details": True,
# 	"label_details": True,
# 	"interpolated_frames": True
#   })

# results = labels.result

# with open(ndjson_output, 'w') as f:
# 	for i in results:
# 		f.write(f"{json.dumps(i)}\n")

with open(ndjson_output, 'r') as f:
	data = ndjson.load(f)

annot_list = []
for row in data:
	img_id = row['data_row']['external_id']
	img_w = row['media_attributes']['width']
	img_h = row['media_attributes']['height']
	labels = row['projects'][f'{PROJECT_ID}']['labels']
	if labels == []:
		continue
	else:
		objects = labels[0]['annotations']['objects']
		for obj in objects:
			class_name = obj['name']
			feature_id = obj['feature_id']
			y1, x1, h, w = tuple(obj['bounding_box'].values())
			x2 = x1 + w
			y2 = y1 + h

			obj_dict = {
				'img_id': img_id,
				'img_h': img_h, 
				'img_w': img_w, 
				'feature_id': feature_id,
				'class_name': class_name,
				'x1': x1,
				'y1': y1, 
				'x2': x2,
				'y2': y2
			}
			annot_list.append(obj_dict)

annot_df = pd.DataFrame.from_dict(annot_list).reset_index(drop=True)
print(annot_df)

annot_df.to_csv(os.path.join(annotation_dir, 'annotations_export.csv'))

print(len(list(annot_df['img_id'].unique())))

