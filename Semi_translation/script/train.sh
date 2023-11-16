
python train.py --name exp_shinkai  --CUT_mode CUT --model semi_cut \ 
 --dataroot ./datasets/unpaired_s2a --paired_dataroot ./datasets/pair_s2a \ 
 --checkpoints_dir ./pretrained_models \
 --dce_idt --lambda_VGG -1  --lambda_NCE_s 0.05 \ 
 --use_curriculum  --gpu_ids 1

##############################################################
: '
usage: train.py 

  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to unpaired images (should have subfolders trainA, trainB) (default: ./datasets/unpaired_s2a)                                                                           
  --paired_dataroot PAIRED_DATAROOT                                                                                                                                                                          
                        path to images (should have subfolders trainA, trainB) (default: ./datasets/pair_s2a)                                                                                                
  --name NAME           name of the experiment. It decides where to store samples and models (default: experiment_name) 

  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
  --checkpoints_dir CHECKPOINTS_DIR
                        models are saved here (default: ./pretrained_models)
  --crop_size CROP_SIZE                                                                                                                                                                             [49/1958]
                        then crop to this size (default: 256)   
  --preprocess PREPROCESS
                        scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none] (default:
                        resize_and_crop)
  --epoch EPOCH         which epoch to load? set to latest to use latest cached model (default: latest)
  --verbose             if specified, print more debugging information (default: False)
  --continue_train      continue training: load the latest model (default: False)
  --epoch_count EPOCH_COUNT
                        the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ... (default: 1)
  --phase PHASE         train, val, test, etc (default: train)

  --CUT_mode            choices=(CUT, cut, FastCUT, fastcut) (default: CUT)

  --lambda_GAN          weight for GAN loss：GAN(G(X)) (default: 1.0)
  --lambda_GAN_p        weight for supervised GAN loss：GAN(G(X)) (default: 1.0)
  --lambda_HDCE         weight for HDCE loss: HDCE(G(X), X) (default: 0.1)
  --lambda_SRC          weight for SRC loss: SRC(G(X), X) (default: 0.05)
  --lambda_NCE_s        weight for StylePatchNCE loss: NCE(G(X^p), Y^p) (default: 0.1)
  --lambda_VGG          weight for VGG content loss: VGG(G(X), Y) (default: 0.1)
  --isDecay            gradually decrease the weight for the supervised training branch (default: True)
        
'
