# Segmentation Selection
```
cd Seg_selection
```

### Installation
Please install dependencies by
```bash 
conda env create -f Seg_selection/environment.yml
conda activate stylegan-mask2former
```
- Sorry that the environment package file may contain additional packages that are not essential.

This code relies on the Mask2Former repo. To set up, follow these steps:
```bash
cd Seg_selection/

# conda environment setup
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
- For more detailed instructions please refer to [Mask2Former installation instructions](https://github.com/facebookresearch/Mask2Former.git).


### Pretrained Models Preparation
1. Download the pretrained StyleGAN2 checkpoint files as follows:
    ```
    cd ../Pseudo_generation
    wget https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.0.1/lhq-220000.pt -P checkpoints
    wget https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.0.1/shinkai-221000.pt -P checkpoints
    ```

2. Download the pretrained Segmentation models
    
    We use the following configuration for semantic segmentation: [ade20k-swin-base-config](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_384_bs16_160k_res640.yaml). You can download the corresponding checkpoint using following script:
    ```
    cd ../Seg_selection
    wget -P pretrained_Mask2Former/ https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_384_bs16_160k_res640/model_final_503e96.pkl
    ```

### Generate pseudo-paired samples!
Before running following command, ensure that you add the directory to your [`sys.path`](https://github.com/Yuxinn-J/Scenimefy/blob/0a8d9872431dc914d06a1abf77609cf706c4b496/Seg_selection/generate_pair.py#L2) within the script to import `Generator` module from different directory.
```
python generate_pair.py
```
- arguments for customization:

    ```
    --truncation: Truncation ratio (default: 0.7).
    --ckpt1: Path to the original model checkpoint.
    --ckpt2: Path to the finetuned model checkpoint.
    --num_sample: Number of paired samples to be generated (default: 30).
    --output_path: Path to save the paired sample images (default: "./data/s2a_shinkai").
    --seg_loss_th: Threshold of segmentation loss for semantic consistency (default: 5.0).
    --seg_cat_th: Threshold of detected category for semantic abundance (default: 1).
    ```
