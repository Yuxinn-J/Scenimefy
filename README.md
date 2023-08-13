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

[Paper](https://arxiv.org/abs/) | [Project Page](https://yuxinn-j.github.io/projects/Scenimefy.html) | [Anime Scene Dataset](#anime-scene-dataset) | ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Yuxinn-J/Scenimefy)

  </br>
  
  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/teaser.png">
  </div>

</div>

-----------------------------------------------------

**Changelog**

- [TODO] Code release.
- [08/2023] Dataset is released.
- [08/2023] Project page is built.
- [07/2023] The paper is accepted to ICCV 2023 :tada:!
-----------------------------------------------------

## Getting Started

## Anime Scene Dataset
![anime-dataset](assets/anime-dataset.png)
It is a high-quality anime scene dataset comprising 5,958 images with the following features:
- High-resolution (1080×1080)
- Shinkai-style (from 9 Mokota Shinkai films)
- Pure anime scene: manual dataset curation by eliminating irrelevant and low-quality images

In compliance with copyright regulations, we cannot directly release the anime images. However, you can conveniently prepare the dataset following instructions [here](anime_dataset/README.md).

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jiang2023scenimefy,
  title={Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation},
  author={Jiang, Yuxin and Jiang, Liming and Yang, Shuai and and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```

## Acknowledgments

The code is mainly developed based on [FreezeG](https://github.com/bryandlee/FreezeG) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC). We thank facebook for their contribution of [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## License

Distributed under the S-Lab License. See [LICENSE.md](./LICENSE.md) for more information.
