set -ex

python test.py --name shinkai-test  --dataroot ./datasets/Sample --results_dir ./results/ \
--CUT_mode CUT  --model cut --phase test --epoch Shinkai --preprocess none

##############################################################
: '
usage: test.py 

  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to images (should have subfolders trainA, trainB, valA, valB, etc) (default: ./datasets/Sample)
  --name NAME           name of the experiment. It decides where to store samples and models (default: experiment_name)

  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
  --checkpoints_dir CHECKPOINTS_DIR
                        models are saved here (default: ./pretrained_models)
  --preprocess PREPROCESS
                        scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none] (default:
                        resize_and_crop)
  --epoch EPOCH         which epoch to load? set to latest to use latest cached model (default: latest)
  --verbose             if specified, print more debugging information (default: False)
  --results_dir RESULTS_DIR
                        saves results here. (default: ./results/)
  --phase PHASE         train, val, test, etc (default: test)
  --num_test NUM_TEST   how many test images to run (default: 1000)
'