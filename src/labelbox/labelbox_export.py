import labelbox as lb
from _labelbox_config import LB_API_KEY, PROJECT_ID, DATASET_ID
import json
import os
import pandas as pd

import ndjson

annotation_dir = '../../data/annotations/'

ndjson_output = os.path.join(annotation_dir, 'annotations_export.ndjson')
# ndjson_output = os.path.join(annotation_dir, 'export-result.ndjson')

client = lb.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)


params = {
	"data_row_details": True,
	"attachments": False,
	"project_details": True,
	"performance_details": False,
	"label_details": True,
	"interpolated_frames": False
}

export_task = project.export_v2(
	params=params
	)
export_task.wait_till_done()

if export_task.errors:
  print(export_task.errors)

export_json = export_task.result
print("results: ", export_json)

with open(ndjson_output, 'w') as f:
	for i in export_json:
		
		f.write(f"{json.dumps(i)}\n")

