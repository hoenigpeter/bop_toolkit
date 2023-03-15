# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition

"""Calculates Instance Mask Annotations in Coco Format."""

import numpy as np
import os
import datetime
import json

from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'ycbv',

  # Dataset split. Options: 'train', 'test'.
  'dataset_split': 'train',

  # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
  'dataset_split_type': 'pbr',

  # bbox type. Options: 'modal', 'amodal'.
  'bbox_type': 'amodal',

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  'scene_paths': os.path.join(config.datasets_path, '{dataset}', '{split}',  '{dataset_split_type}', '{scene_id:06d}'),
}
################################################################################

datasets_path = p['datasets_path']
dataset_name = p['dataset']
split = p['dataset_split']
split_type = p['dataset_split_type']
bbox_type = p['bbox_type']

dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

complete_split = split
if dp_split['split_type'] is not None:
    complete_split += '_' + dp_split['split_type']

CATEGORIES = [{'id': obj_id, 'name':str(obj_id), 'supercategory': dataset_name} for obj_id in dp_model['obj_ids']]
INFO = {
    "description": dataset_name + '_' + split,
    "url": "https://github.com/thodan/bop_toolkit",
    "version": "0.1.0",
    "year": datetime.date.today().year,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

for scene_id in dp_split['scene_ids']:
    segmentation_id = 1

    coco_scene_output = {
        "info": INFO,
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # Load info about the GT poses (e.g. visibility) for the current scene.
    scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(scene_id=scene_id))
    scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id), keys_to_int=True)
    # Output coco path
    coco_gt_path = dp_split['scene_gt_coco_tpath'].format(scene_id=scene_id)
    if bbox_type == 'modal':
        coco_gt_path = coco_gt_path.replace('scene_gt_coco', 'scene_gt_coco_modal')
    misc.log('Calculating Coco Annotations - dataset: {} ({}, {}), scene: {}'.format(
          p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id))
    
    # Go through each view in scene_gt
    for scene_view, inst_list in scene_gt.items():
        im_id = int(scene_view)
        
        img_path = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
        shutil.copy(img_path, os.path.join(dst.))

        print(dp_split)
        print(dp_split['split_path'])
        print('{:06d}'.format(scene_id))
        
