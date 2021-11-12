#!/usr/bin/env python3.7

import argparse
import os
import numpy as np
import json
from pycocotools import mask
from skimage import measure

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--out", required=True, help = "path of output directory")
	parser.add_argument("--in", required=True, help = "path of wgisd directory")
	args = vars(parser.parse_args())

	out_dir = args["out"]
	wgisd_dir = args["in"]

	if not os.path.isdir(wgisd_dir):
		print("error with wgisd path")
		exit(1)
	if not os.path.isdir(out_dir):
	    os.makedirs(out_dir)

	os.chdir(wgisd_dir) #"/home/andreaceruti/Desktop/wgisd_instance_segmentation"

	#take the image instances in a list
	instances = []
	for dirname, dirnames, filenames in os.walk('images'):
	        for filename in [f for f in filenames if f.endswith('.jpg')]:
	                    instances.append(filename[:-4])

	#initialize the annotations file
	json_annotations = {

				"licenses":[
					{
						"name": "",
						"id": 0,
						"url": ""
					}
				],
				"info": {
					"contributor": "",
					"date_created": "",
					"description": "",
					"url": "",
					"version": "",
					"year": ""
				},
				"categories":[
					{
						"id": 1,
						"name": "grape bunch", 
						"supercategory": ""
					}
				],
	          	"images": [],
	          	"annotations": []
	                    }

	image_count = 1
	annotations_count = 1

	#add images and masks to image array
	for filename in instances:
		annotation_file = os.path.join(wgisd_dir + '/masks', filename + '.npz')
		if os.path.isfile(annotation_file):
			annotation_mask = np.load(annotation_file)['arr_0'].astype(np.uint8)

        #in mask we have n binary masks for each image, one for each grape bunch instance, so the dimension will be (Height, width, n-cluster)

		json_image_object = {
			"id": image_count,
			"width": annotation_mask.shape[1],
			"height": annotation_mask.shape[0],
			"file_name": filename + '.jpg',
			"license": 0,
			"flickr_url": "",
			"coco_url": "",
			"date_captured": 0
		}

		for i in range(annotation_mask.shape[2]): #for each binary mask, save the annotation object
			
			ground_truth_binary_mask = np.array(annotation_mask[:,:,i], dtype = np.uint8)

			fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
			encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
			ground_truth_area = mask.area(encoded_ground_truth)
			ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
			contours = measure.find_contours(ground_truth_binary_mask, 0.5)

			json_annotation_object = {
				"segmentation": [],
        		"area": ground_truth_area.tolist(),
        		"iscrowd": 0,
        		"image_id": image_count,
        		"bbox": ground_truth_bounding_box.tolist(),
        		"category_id": 1,
        		"id": annotations_count
			}

			annotations_count += 1

			for contour in contours:
				contour = np.flip(contour, axis=1)
				segmentation = contour.ravel().tolist()
				json_annotation_object["segmentation"].append(segmentation)

			json_annotations["annotations"].append(json_annotation_object)

		json_annotations["images"].append(json_image_object)
		image_count += 1
	


	#write json files to output directory           
	os.chdir(out_dir)

	with open('data.json', 'w') as f:
	    json.dump(json_annotations, f, ensure_ascii=False, indent=4)                    
