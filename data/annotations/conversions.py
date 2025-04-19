import pandas as pd
import ndjson
import json
import numpy as np

class COCOFormat():

	def __init__(self):
		self.json = {}

		self.json['info'] = {}
		self.json['licenses'] = []
		self.json['images'] = []
		self.json['annotations'] = []
		self.json['categories'] = []
		self.max_license = 1

	def _update_info(self, description, url, version, year, contributor, date_created):
		"""
		Update the info value
		"""
		self.json['info']['description'] = description
		self.json['info']['url'] = url
		self.json['info']['version'] = version
		self.json['info']['year'] = year
		self.json['info']['contributor'] = contributor
		self.json['info']['date_created'] = date_created

	def _update_license(self, url, name):
		"""
		Update the info value
		"""

		if len(self.json['licenses'])==0:
			ID = self.max_license
		else:
			ID = np.max([item['id'] for item in self.json['licenses']]) + 1
			if ID > self.max_license:
				self.max_license += 1

		self.json['licenses'].append(
			{
				"url": url,
				"id": ID,
				"name": name
			}
		)

	def _add_category(self, cat_id, name, supercategory):
		"""
		Add one category to the json 'categories' key list
		"""

		assert cat_id not in [item['id'] for item in self.json['categories']]

		self.json['categories'].append(
			{
				"id": int(cat_id),
				"name": name,
				"supercategory": supercategory
			}
		)

	def _add_image(self, img_id, width, height, file_name, license, date_captured):
		"""
		Add one image to the 'images' key list
		"""

		assert license <= self.max_license

		self.json['images'].append(
			{
				"id": img_id,
				"width": width,
				"height": height,
				"file_name": file_name,
				"license": license,
				"date_captured": date_captured
			}
		)

	def _add_annotation(self, annot_id, img_id, cat_id, area, bbox, license):
		"""
		"""

		assert cat_id in [item['id'] for item in self.json['categories']]
		assert license <= self.max_license

		self.json['annotations'].append(
			{
				"id": annot_id,
				"image_id": img_id,
				"category_id": cat_id,
				"area": area,
				"bbox": bbox,
				"license": license
			}
		)




def ndjson_to_coco(ndjson_file: str, output_file: str, categories: dict) -> None:
	"""
	Takes a .ndjson file from labelbox, scrubs it clean of any of the sensitive information regarding project IDs, email addresses, etc
	and converts it to standard COCO format

	Arguments
	ndjson_file: a Python string of the absolute or relative path to the exported ndjson file
	output_file: a Python string of the absolute or relative path for the output COCO json file

	"""

	def _extract_key(d, val):
		return int([k for k, v in d.items() if v == val][0])

	with open(ndjson_file, 'r') as f:
		raw_data = ndjson.load(f)
	# initialize COCO object
	coco_file = COCOFormat()

	# add COCO object info
	coco_file._update_info(description='Annotations for the sainfoin seed image dataset 10.5281/zenodo.8346924',
		url='https://zenodo.org/doi/10.5281/zenodo.8346923',
		version='0.1.0',
		year=2023,
		contributor='Bo Meyering',
		date_created='2023-09-19')

	# Add licensing
	coco_file._update_license(url="https://creativecommons.org/licenses/by-sa/4.0/", name="Attribution-ShareAlike 4.0 International License")
	
	# Add in all classes and categories
	for k,v in categories.items():
		coco_file._add_category(k, v, "")

	# Loop through all images and annotations
	for file in raw_data:
		data_row = file['data_row']
		attributes = file['media_attributes']
		projects = file['projects']

		img_id = data_row['id']
		file_name = data_row['external_id'].replace('JPG', 'jpg')
		width = attributes['width']
		height = attributes['height']
		date_captured = "2023-07-07"
		license = 1

		# update image_list
		coco_file._add_image(img_id, width, height, file_name, license, date_captured)

		for i in projects.keys():
			annotations = projects[i]
			# print(annotations)
			# print("\n\n")

			objects = annotations['labels'][0]['annotations']['objects']
			for obj in objects:
				# print(obj)
				annot_id = obj['feature_id']
				cat_id = _extract_key(categories, obj['name'])
				# print(cat_id)
				bbox_dict = obj['bounding_box']
				bbox = [bbox_dict['left'], bbox_dict['top'], bbox_dict['width'], bbox_dict['height']]
				area = bbox[2]*bbox[3]
				license=1
				coco_file._add_annotation(annot_id, img_id, cat_id, area, bbox, license)

	with open(output_file, 'w') as f:
		json.dump(coco_file.json, f)
	
	# print(len(coco_file.json['annotations']))
	# for i in range(1):
	# 	print(data[i]['data_row'])
	# 	print(data[i].keys())

	return None



classes = {
    '0': 'background',
    '1': 'split',
    '2': 'seed',
    '3': 'pod'
}

ndjson_to_coco('export-result.ndjson', 'coco_annotations.json', categories=classes)

