import labelbox as lb
from labelbox import DataRow
from _labelbox_config import LB_API_KEY, PROJECT_ID, DATASET_ID
import json
import os
import pandas as pd

import ndjson

annotation_dir = '../../data/annotations/'

ndjson_output = os.path.join(annotation_dir, 'annotations_export.ndjson')

print("Establishing connection to Labelbox")
client = lb.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)
# print(project.result)



dataset = client.get_dataset(DATASET_ID)
global_keys = [data_row.global_key for data_row in list(dataset.data_rows())]
# print(global_keys)

res = {}
with open('export_results.ndjson', 'w') as b:
	for i, key in enumerate(global_keys):
		params = {
			"data_row_details": True,
			"project_details": True,
			"label_details": True,
		}

		filters = {
		"global_keys": [key]
		}

		export_task = project.export_v2(
			params=params,
			filters=filters
			)
		export_task.wait_till_done()

		if export_task.errors:
		  print(export_task.errors)

		export_json = export_task.result
		n_objects = len(export_json[0]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'])
		print(f"Global Key: {key} \t {i}")
		print(f"N Objects: {n_objects}\n")
		res[key] = n_objects

		# print(f"results: {export_json}\n\n")

		# with open(ndjson_output, 'w') as f:
		# 	for i in export_json:
				
		# 		f.write(f"{json.dumps(i)}\n")
		# f.writelines([f"{key}\t{n_objects}"])
		b.write(f"{json.dumps(export_json[0])}\n")
	# print(res)

# for data_row in list(dataset.data_rows()):
# 	print(data_row.uid)
# 	export_task = DataRow.export_v2(
# 		client=client,
# 		data_rows=[data_row.uid],
# 		params={
#     	"data_row_details": True,
#     	"performance_details": False,
#     	"label_details": True
#     	}
# 	)
# 	export_task.wait_till_done()
# 	print(export_task.result)
# print(list(dataset.data_rows()))
# export_task = DataRow.export_v2(
# 	client=client,
# 	data_rows=[data_row.uid for data_row in list(dataset.data_rows())],

#             # or a list of DataRow objects: data_rows = data_set.data_rows.list()

#             # or a list of global_keys=["global_key_1", "global_key_2"],

#             # Note that exactly one of: data_rows or global_keys parameters can be passed in at a time

#             # and if data rows ids is present, global keys will be ignored
#     params={
#     "data_row_details": True,
#     "performance_details": False,
#     "label_details": True
#     }
# )

# export_task.wait_till_done()

# export_task.result




# params = {
# 	"data_row_details": True,
# 	"attachments": False,
# 	"project_details": True,
# 	"performance_details": False,
# 	"label_details": True,
# 	"interpolated_frames": False
# }

# export_task = project.export_v2(
# 	params=params
# 	)
# export_task.wait_till_done()

# if export_task.errors:
#   print(export_task.errors)

# export_json = export_task.result
# print("results: ", export_json)

# with open(ndjson_output, 'w') as f:
# 	for i in export_json:
		
# 		f.write(f"{json.dumps(i)}\n")