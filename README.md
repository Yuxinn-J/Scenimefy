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

[Paper](https://arxiv.org/abs/) | [Project Page](https://yuxinn-j.github.io/projects/Scenimefy.html) | [Anime Scene Dataset](#open_file_folder-anime-scene-dataset) 
<!-- | ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Yuxinn-J/Scenimefy) -->

  </br>
  
  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/teaser.png">
  </div>

</div>

-----------------------------------------------------

### Updates
- [TODO] Full training pipeline release.
- [TODO] Training code release.
- [08/2023] Inference code is released.
- [08/2023] Dataset is released.
- [08/2023] Project page is built.
- [07/2023] The paper is accepted to ICCV 2023 :partying_face:!
-----------------------------------------------------

## :wrench: Installation
1. Clone this repo:
    ```bash
    git clone https://github.com/Yuxinn-J/Scenimefy.git
    cd Scenimefy
    ```
2. Install dependent packages:
  after installing [Anaconda](https://www.anaconda.com/), you can create a new Conda environment using `conda env create -f Semi_translation/environment.yml`.

## :zap: Quick Inference
- Download pre-trained models: [Shinkai_net_G.pth](https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.1.0/Shinkai_net_G.pth)
  ```bash
  wget https://github.com/Yuxinn-J/Scenimefy/releases/download/v0.1.0/Shinkai_net_G.pth -P Semi_translation/pretrained_models/shinkai-test/
  ```

- Inference! Simply run the following command, or refer the [`./Semi_translation/script/test.sh`](./Semi_translation/script/test.sh) for detailed usage:
  ```bash
  cd Semi_translation
  
  python test.py --name shinkai-test --CUT_mode CUT  --model cut --phase test --epoch Shinkai --preprocess none
  ```
  - Results are in the `./Semi_translation/results/shinkai-test/` by default. 
  - To prepare your own test images, you can refer to the data folder structure in [`./Semi_translation/datasets/Sample`](./Semi_translation/datasets/Sample), and place your test images in `testA`. 

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
  author={Jiang, Yuxin and Jiang, Liming and Yang, Shuai and and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```

## :hugs: Acknowledgments

Our code is mainly developed based on [FreezeG](https://github.com/bryandlee/FreezeG) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC)(single-modal image translation setting). We thank facebook for their contribution of [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## :newspaper_roll: License

Distributed under the S-Lab License. See [LICENSE.md](./LICENSE.md) for more information.
