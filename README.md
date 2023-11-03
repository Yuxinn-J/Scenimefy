<div align="center">

  <h1>Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation</h1>
  
  <div>
      <a href="https://yuxinn-j.github.io/" target="_blank">Yuxin Jiang</a><sup>*</sup>,
      <a href="https://liming-jiang.com/" target="_blank">Liming Jiang</a><sup>*</sup>,
      <a href="https://williamyang1991.github.io/" target="_blank">Shuai Yang</a>,
      <a href="https://www.mmlab-ntu.com/person/ccloy/" target="_blank">Chen Change Loy</a>
  </div>
  <div>
      MMLab@NTU affiliated with S-Lab, Nanyang Technological University
  </div>
  <div>
  In ICCV 2023.
  </div>

:page_with_curl:[**Paper**](https://arxiv.org/abs/2308.12968) **|** :globe_with_meridians:[**Project Page**](https://yuxinn-j.github.io/projects/Scenimefy.html) **|** :open_file_folder:[**Anime Scene Dataset**](#open_file_folder-anime-scene-dataset) **|** 🤗[**Demo**](https://huggingface.co/spaces/YuxinJ/Scenimefy)


  </br>
  
  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/teaser.png">
  </div>

</div>

-----------------------------------------------------

### Updates

- [11/2023] Train code is available.
- [08/2023] Integrated to [Hugging Face](https://huggingface.co/spaces/YuxinJ/Scenimefy). Enjoy the web demo!
- [08/2023] Inference code and Dataset is released.
- [08/2023] Project page is built.
- [07/2023] The paper is accepted to ICCV 2023!
-----------------------------------------------------

## :wrench: Installation
1. Clone this repo:
    ```bash
    git clone https://github.com/Yuxinn-J/Scenimefy.git
    cd Scenimefy
    ```
2. Install dependent packages:
  After installing [Anaconda](https://www.anaconda.com/), create a new Conda environment using `conda env create -f Semi_translation/environment.yml`.

## :zap: Quick Inference
1. [Python script](#python-script)  2. [Gradio demo](#gradio-demo) 
### Python script
- Download pre-trained models: [Shinkai_net_G.pth](https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.1.0/Shinkai_net_G.pth)
  ```bash
  wget https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.1.0/Shinkai_net_G.pth -P Semi_translation/pretrained_models/shinkai-test/
  ```

- Inference! Simply run the following command, or refer the [`./Semi_translation/script/test.sh`](./Semi_translation/script/test.sh) for detailed usage:
  ```bash
  cd Semi_translation
  
  python test.py --dataroot ./datasets/Sample --name shinkai-test --CUT_mode CUT  --model cut --phase test --epoch Shinkai --preprocess none
  ```
  - Results will be saved in `./Semi_translation/results/shinkai-test/` by default. 
  - To prepare your own test images, you can refer to the data folder structure in [`./Semi_translation/datasets/Sample`](./Semi_translation/datasets/Sample), and place your test images in `testA`. 

### Gradio demo
- We provide a UI for testing Scenimefy, which is built with gradio. To launch the demo, simply execute the following command in your terminal:
  ```
  git clone https://huggingface.co/spaces/YuxinJ/Scenimefy
  pip install -r requirements.txt
  pip install gradio
  python app.py
  ```
- This demo is also hosted on [Hugging Face🤗](https://huggingface.co/spaces/YuxinJ/Scenimefy).

## :train: Quick I2I Train
### Dataset Preparation
- **[LHQ dataset](https://github.com/universome/alis#lhq-dataset)**: a dataset of 90,000 nature landscape images [[downlaod link](https://disk.yandex.ru/d/HPEEntpLv8homg)]. Place it in `./datasets/unpaired_s2a`, and rename as `trainA`.
- **Anime dataset**: 5,958 shinkai-style anime scene images. Please follow the instructions in [`Anime_dataset/README.md`](Anime_dataset/README.md). Place it in `./datasets/unpaired_s2a`, and rename as `trainB`.
- **Pseudo-paired dataset**: 30,000 synthetic pseudo paired images generated from StyleGAN with the same seed. You may finetune your own StyleGAN or use our provided data [[downlaod link](https://entuedu-my.sharepoint.com/:u:/g/personal/c200203_e_ntu_edu_sg/EaZ-8U_3HbBKh9qq4AfJWmUByIuPwn_3GEDpPc84GXuU7w?e=gAs850)] for quick start. Place them in `./datasets/pair_s2a`
- Create your own dataset

### Training
Refer to the [`./Semi_translation/script/train.sh`](./Semi_translation/script/train.sh) file, or use the following command:
  ```
  python train.py --name exp_shinkai  --CUT_mode CUT --model semi_cut \ 
  --dataroot ./datasets/unpaired_s2a --paired_dataroot ./datasets/pair_s2a \ 
  --checkpoints_dir ./pretrained_models \
  --dce_idt --lambda_VGG -1  --lambda_NCE_s 0.05 \ 
  --use_curriculum  --gpu_ids 0
  ```
  - If the anime dataset quality is low, consider add a global perceptual loss to maintain content consistency, e.g., set `--lambda_VGG 0.2`.

## :checkered_flag: Start From Scratch
### StyleGAN Finetuning [TODO]
- Follow the instructions in [`Pseudo_generation/README.md`](Pseudo_generation/README.md).
### Segmenation Selection
- Follow the instructions in [`Seg_selection/README.md`](Seg_selection/README.md).

## :open_file_folder: Anime Scene Dataset
![anime-dataset](assets/anime-dataset.png)
It is a high-quality anime scene dataset comprising 5,958 images with the following features:
- High-resolution (1080×1080)
- Shinkai-style (from 9 Mokota Shinkai films)
- Pure anime scene: manual dataset curation by eliminating irrelevant and low-quality images

In compliance with copyright regulations, we cannot directly release the anime images. However, you can conveniently prepare the dataset following instructions [here](Anime_dataset/README.md).

## :love_you_gesture: Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jiang2023scenimefy,
  title={Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation},
  author={Jiang, Yuxin and Jiang, Liming and Yang, Shuai and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```

## :hugs: Acknowledgments

Our code is mainly developed based on [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC). We thank facebook for their contribution of [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## :newspaper_roll: License

Distributed under the S-Lab License. See [LICENSE.md](./LICENSE.md) for more information.
