# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in pose estimates saved in the BOP format."""

import os
import numpy as np
import itertools

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization

import cv2

def compute_bounding_box_corners(obj_center, size):
    min_x, min_y, min_z = obj_center
    size_x, size_y, size_z = size

    corners = np.array([
        [min_x - size_x / 2, min_y - size_y / 2, min_z - size_z / 2],  # Front bottom left
        [min_x + size_x / 2, min_y - size_y / 2, min_z - size_z / 2],  # Front bottom right
        [min_x + size_x / 2, min_y + size_y / 2, min_z - size_z / 2],  # Front top right
        [min_x - size_x / 2, min_y + size_y / 2, min_z - size_z / 2],  # Front top left
        [min_x - size_x / 2, min_y - size_y / 2, min_z + size_z / 2],  # Back bottom left
        [min_x + size_x / 2, min_y - size_y / 2, min_z + size_z / 2],  # Back bottom right
        [min_x + size_x / 2, min_y + size_y / 2, min_z + size_z / 2],  # Back top right
        [min_x - size_x / 2, min_y + size_y / 2, min_z + size_z / 2]   # Back top left
    ])

    return corners

# PARAMETERS.
################################################################################
p = {
  # Top N pose estimates (with the highest score) to be visualized for each
  # object in each image.
  'n_top': 1,  # 0 = all estimates, -1 = given by the number of GT poses.

  # True = one visualization for each (im_id, obj_id), False = one per im_id.
  'vis_per_obj_id': True,

  # Indicates whether to render RGB image.
  'vis_rgb': True,

  # Indicates whether to resolve visibility in the rendered RGB images (using
  # depth renderings). If True, only the part of object surface, which is not
  # occluded by any other modeled object, is visible. If False, RGB renderings
  # of individual objects are blended together.
  'vis_rgb_resolve_visib': True,

  # Indicates whether to render depth image.
  'vis_depth_diff': False,

  # If to use the original model color.
  'vis_orig_color': True,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'cpp',  # Options: 'vispy', 'cpp', 'python'.

  # Names of files with pose estimates to visualize (assumed to be stored in
  # folder config.eval_path). See docs/bop_challenge_2019.md for a description
  # of the format. Example results can be found at:
  # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/
  'result_filenames': [
    '/home/hoenig/bop_toolkit/scripts/pix2pose-iccv19_lmo-test.csv',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # Folder for output visualisations.
  'vis_path': os.path.join(config.output_path, 'vis_est_poses'),

  # Path templates for output images.
  'vis_rgb_tpath': os.path.join(
    '{vis_path}', '{result_name}', '{scene_id:06d}', '{vis_name}.jpg'),
  'vis_depth_diff_tpath': os.path.join(
    '{vis_path}', '{result_name}', '{scene_id:06d}',
    '{vis_name}_depth_diff.jpg'),
}
################################################################################


# Load colors.
colors_path = os.path.join(
  os.path.dirname(visualization.__file__), 'colors.json')
colors = inout.load_json(colors_path)

for result_fname in p['result_filenames']:
  misc.log('Processing: ' + result_fname)

  # Parse info about the method and the dataset from the filename.
  result_name = os.path.splitext(os.path.basename(result_fname))[0]
  result_info = result_name.split('_')
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'eval'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

  # Rendering mode.
  renderer_modalities = []
  if p['vis_rgb']:
    renderer_modalities.append('rgb')
  if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
    renderer_modalities.append('depth')
  renderer_mode = '+'.join(renderer_modalities)

  # Create a renderer.
  width, height = dp_split['im_size']
  ren = renderer.create_renderer(
    width, height, p['renderer_type'], mode=renderer_mode)

  # Load object models.
  models = {}
  for obj_id in dp_model['obj_ids']:
    misc.log('Loading 3D model of object {}...'.format(obj_id))
    model_path = dp_model['model_tpath'].format(obj_id=obj_id)
    model_color = None
    if not p['vis_orig_color']:
      model_color = tuple(colors[(obj_id - 1) % len(colors)])
    print(obj_id)
    print(model_path)
    print(model_color)
    ren.add_object(obj_id, model_path, surf_color=model_color)

  models_info = inout.load_json(dp_model['models_info_path'], keys_to_int=True)
  print(models_info)
  # Load pose estimates.
  misc.log('Loading pose estimates...')
  ests = inout.load_bop_results(
    os.path.join(config.results_path, result_fname))

  # Organize the pose estimates by scene, image and object.
  misc.log('Organizing pose estimates...')
  ests_org = {}
  for est in ests:
    ests_org.setdefault(est['scene_id'], {}).setdefault(
      est['im_id'], {}).setdefault(est['obj_id'], []).append(est)

  for scene_id, scene_ests in ests_org.items():

    # Load info and ground-truth poses for the current scene.
    scene_camera = inout.load_scene_camera(
      dp_split['scene_camera_tpath'].format(scene_id=scene_id))
    # scene_gt = inout.load_scene_gt(
    #   dp_split['scene_gt_tpath'].format(scene_id=scene_id))

    for im_ind, (im_id, im_ests) in enumerate(scene_ests.items()):

      if im_ind % 10 == 0:
        split_type_str = ' - ' + split_type if split_type is not None else ''
        misc.log(
          'Visualizing pose estimates - method: {}, dataset: {}{}, scene: {}, '
          'im: {}'.format(method, dataset, split_type_str, scene_id, im_id))

      # Intrinsic camera matrix.
      K = scene_camera[im_id]['cam_K']

      im_ests_vis = []
      im_ests_vis_obj_ids = []
      for obj_id, obj_ests in im_ests.items():

        # Sort the estimates by score (in descending order).
        obj_ests_sorted = sorted(
          obj_ests, key=lambda est: est['score'], reverse=True)

        # Select the number of top estimated poses to visualize.
        if p['n_top'] == 0:  # All estimates are considered.
          n_top_curr = None
        # elif p['n_top'] == -1:  # Given by the number of GT poses.
        #   n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
        #   n_top_curr = n_gt
        else:  # Specified by the parameter n_top.
          n_top_curr = p['n_top']
        obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

        # Get list of poses to visualize.
        for est in obj_ests_sorted:
          est['obj_id'] = obj_id

          # Text info to write on the image at the pose estimate.
          if p['vis_per_obj_id']:
            est['text_info'] = [
              {'name': '', 'val': est['score'], 'fmt': ':.2f'}
            ]
          else:
            val = '{}:{:.2f}'.format(obj_id, est['score'])
            est['text_info'] = [{'name': '', 'val': val, 'fmt': ''}]

        im_ests_vis.append(obj_ests_sorted)
        im_ests_vis_obj_ids.append(obj_id)

      # Join the per-object estimates if only one visualization is to be made.
      if not p['vis_per_obj_id']:
        im_ests_vis = [list(itertools.chain.from_iterable(im_ests_vis))]

      for ests_vis_id, ests_vis in enumerate(im_ests_vis):

        # Load the color and depth images and prepare images for rendering.
        rgb = None
        if p['vis_rgb']:
          if 'rgb' in dp_split['im_modalities']:
            rgb = inout.load_im(dp_split['rgb_tpath'].format(
              scene_id=scene_id, im_id=im_id))[:, :, :3]
          elif 'gray' in dp_split['im_modalities']:
            gray = inout.load_im(dp_split['gray_tpath'].format(
              scene_id=scene_id, im_id=im_id))
            rgb = np.dstack([gray, gray, gray])
          else:
            raise ValueError('RGB nor gray images are available.')

        depth = None
        if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
          depth = inout.load_depth(dp_split['depth_tpath'].format(
            scene_id=scene_id, im_id=im_id))
          depth *= scene_camera[im_id]['depth_scale']  # Convert to [mm].

        # Visualization name.
        if p['vis_per_obj_id']:
          vis_name = '{im_id:06d}_{obj_id:06d}'.format(
            im_id=im_id, obj_id=im_ests_vis_obj_ids[ests_vis_id])
        else:
          vis_name = '{im_id:06d}'.format(im_id=im_id)

        # Path to the output RGB visualization.
        vis_rgb_path = None
        if p['vis_rgb']:
          vis_rgb_path = p['vis_rgb_tpath'].format(
            vis_path=p['vis_path'], result_name=result_name, scene_id=scene_id,
            vis_name=vis_name)

        # Path to the output depth difference visualization.
        vis_depth_diff_path = None
        if p['vis_depth_diff']:
          vis_depth_diff_path = p['vis_depth_diff_tpath'].format(
            vis_path=p['vis_path'], result_name=result_name, scene_id=scene_id,
            vis_name=vis_name)

        for pose in ests_vis:
            print("im_id: ", im_id)
            print("obj_id: ", obj_id)
            print(models_info[obj_id]['min_x'])
            print(ests_vis)
            min_point = (models_info[pose['obj_id']]['min_x'], models_info[pose['obj_id']]['min_y'], models_info[pose['obj_id']]['min_z'])
            size = (models_info[pose['obj_id']]['size_x'], models_info[pose['obj_id']]['size_y'], models_info[pose['obj_id']]['size_z'])
            print(size)
            print(min_point)
            obj_center = np.array(min_point) + 0.5 * np.array(size)
            corners = compute_bounding_box_corners(obj_center, size)

            corner_points = compute_bounding_box_corners(obj_center, size)
            corner_points_3d = misc.transform_pts_Rt(corner_points, pose['R'], pose['t'])

            # Project 3D corner points onto the image plane
            corner_points_2d = np.dot(K, corner_points_3d.T).T
            corner_points_2d = corner_points_2d[:, :2] / corner_points_2d[:, 2:]

            print("3D Bounding Box Corners:")
            print(corners)
            print("Transformed Corners:")
            print(corner_points)
            print("Projected 2D Corners:")
            print(corner_points_2d)

            # Define lines connecting the corner points to form edges of the bounding box
            lines = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting lines
            ]

            # Draw the lines on the RGB image
            for line in lines:
                pt1 = tuple(corner_points_2d[line[0]].astype(int))
                pt2 = tuple(corner_points_2d[line[1]].astype(int))
                cv2.line(rgb, pt1, pt2, (0, 0, 255), 2)  # Red lines

            rgb_image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(vis_rgb_path, rgb_image_rgb)

        # # Visualization.
        # visualization.vis_object_poses(
        #   poses=ests_vis, K=K, renderer=ren, rgb=rgb, depth=depth,
        #   vis_rgb_path=vis_rgb_path, vis_depth_diff_path=vis_depth_diff_path,
        #   vis_rgb_resolve_visib=p['vis_rgb_resolve_visib'])
        
misc.log('Done.')
