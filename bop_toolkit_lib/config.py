# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  #datasets_path = r'/path/to/bop/datasets'
  #datasets_path = '/home/hoenig/BOP/gdrnpp_bop2022/datasets/BOP_DATASETS'
  #datasets_path = '/home/hoenig/temp/gdrnet_pipeline/src/gdrnpp/datasets/BOP_DATASETS'
  #datasets_path = '/home/hoenig/BOP/Pix2Pose/pix2pose_datasets'
  #datasets_path = '/hdd/datasets_bop'
  datasets_path = '/home/hoenig/datasets'

# Folder with pose results to be evaluated.
results_path = r'/home/hoenig/bop_toolkit/scripts'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/home/hoenig/bop_toolkit/scripts/results/eval'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/home/hoenig/bop_toolkit/scripts/results'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/home/hoenig/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
