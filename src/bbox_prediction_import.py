import labelbox as lb
import pandas as pd
import numpy as np
import labelbox.types as lb_types
import uuid

from _labelbox_config import LB_API_KEY, PROJECT_ID

predictions = pd.read_csv('../data/power_analysis/results.csv')
# print(predictions)
predictions['img'] = predictions['img_id'].str.replace('./data/blue_images/', '')
# print(predictions)
img_ids = predictions.img.unique()
# print(img_ids)



PROJECT_ID = 'cll3qlz2202gv07z034gng0xa'


client = lb.Client(api_key=LB_API_KEY)
project = client.get_project(project_id=PROJECT_ID)


project = client.get_project(PROJECT_ID)
labels = project.export_v2(params={
	"data_row_details": True,
	"metadata_fields": False,
	"attachments": False,
	"project_details": True,
	"performance_details": False,
	"label_details": True,
	"interpolated_frames": False
  })
results = labels.result
for i in results:
	external_id = i['data_row']['external_id']
	global_id = i['data_row']['id']
	labels = i['projects'][PROJECT_ID]['labels']
	if (labels == []) & (external_id in img_ids):
		print(external_id)
		preds = predictions.loc[predictions['img'] == external_id]
		preds = preds.loc[preds['score'] >= .95]
		label_list = []
		annotations = []
		for i, row in preds.iterrows():
			bbox_annotation = lb_types.ObjectAnnotation(
				name=row['class'],
				value=lb_types.Rectangle(
					start=lb_types.Point(x=row['xmin'], y=row['ymin']),
					end=lb_types.Point(x=row['xmax'], y=row['ymax'])
				))
			annotations.append(bbox_annotation)

		label_list.append(
			lb_types.Label(data=lb_types.ImageData(uid=global_id),
			annotations=annotations
			))
		upload_job = lb.LabelImport.create_from_objects(
    		client = client, 
    		project_id = PROJECT_ID, 
    		name="label_import_job"+str(uuid.uuid4()),  
    		labels=label_list)

		print(f"Errors: {upload_job.errors}", )
		print(f"Status of uploads: {upload_job.statuses}")



		











# batch = project.create_batch(
#     "image-demo-batch",  # each batch in a project must have a unique name
#     global_keys=[global_key], # paginated collection of data row objects, list of data row ids or global keys
#     priority=1  # priority between 1(highest) - 5(lowest)
# )

# print(f"Batch: {batch}")




# bbox_annotation = lb_types.ObjectAnnotation(
#     name="pod",  # must match your ontology feature"s name
#     value=lb_types.Rectangle(
#         start=lb_types.Point(x=3741, y=2762),  #  x = left, y = top 
#         end=lb_types.Point(x=3809, y=2808),  # x= left + width , y = top + height
#     ))

# label = []
# annotations = [
#     bbox_annotation,
# ]

# print(global_key)

# label.append(
# 	lb_types.Label(data=lb_types.ImageData(uid=global_key),
#                    annotations=annotations))

# print(project.uid)

# # upload_job = lb.MALPredictionImport.create_from_objects(
# #     client=client,
# #     project_id=PROJECT_ID,
# #     name="mal_job" + str(uuid.uuid4()),
# #     predictions=label
# # )
# # upload_job.wait_until_done()


# upload_job = lb.LabelImport.create_from_objects(
#     client = client, 
#     project_id = PROJECT_ID, 
#     name="label_import_job"+str(uuid.uuid4()),  
#     labels=label)

# print(f"Errors: {upload_job.errors}", )
# print(f"Status of uploads: {upload_job.statuses}")

