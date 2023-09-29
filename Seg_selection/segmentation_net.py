import argparse
import glob
import multiprocessing as mp
import os

import torch
from torch import nn
import cv2
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

from Mask2Former.mask2former import add_maskformer2_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_384_bs16_160k_res640.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./pretrained_Mask2Former/model_final_503e96.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


i = 0
def increment_i():
    global i  # global variable
    i += 1    
    return i


class SEMSEG:
    def __init__(self):
        mp.set_start_method("spawn", force=True)
        args = get_parser().parse_args()
        self.cfg = setup_cfg(args)

        self.model = build_model(self.cfg)
        self.model.eval()

        requires_grad(self.model, False)

        if len(self.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        # metadata.stuff_classes
        self.target_class = [
            'building', 'sky', 'tree', 'grass', 'person', 'earth, ground', 'mountain, mount', 
            'plant', 'water', 'house', 'sea'
        ]
    
    def image_prepare(self, image):
        image = torch.squeeze(image) # [C, H, W]
        h, w = image.shape[-2:]
        img = torch.round((image + 1) * 255 / 2) # [-1, 1] -> [0 - 255]
        inputs = {"image": img, "height": h, "width": w}
        return inputs

    def forward_ce(self, fake_img, gt_img, visualize=False):
        """
            Args:
                input_image (Tensor): an image of shape (BS, C, H, W)  (in RGB order) [-1 1].
            Returns:
                cross-entropy loss, number of detected categories
        """
        ce_loss = nn.CrossEntropyLoss()

        f_inputs = self.image_prepare(fake_img)
        g_inputs = self.image_prepare(gt_img)
        
        f_predictions = self.model([f_inputs])[0]
        g_predictions = self.model([g_inputs])[0]

        if visualize == True:
            f_img_bgr = cv2.cvtColor(f_inputs["image"].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            g_img_bgr = cv2.cvtColor(g_inputs["image"].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            g_vis = self.visualize(g_predictions["sem_seg"])
            f_vis = self.visualize(f_predictions["sem_seg"])

            h, w, _ = f_img_bgr.shape
            top_row = np.hstack((g_img_bgr, g_vis))
            bottom_row = np.hstack((f_img_bgr, f_vis))
            grid = np.vstack((top_row, bottom_row))

            # TODO:
            save_dir = "./data/s2a_shinkai/seg_mask"
            Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{save_dir}/{increment_i()}.png", grid)

        f_mask = f_predictions["sem_seg"][None, :]
        g_mask = g_predictions["sem_seg"].argmax(dim=0)[None, :]

        num_seg = g_predictions["sem_seg"].argmax(dim=0).unique().size(dim=0)

        return ce_loss(f_mask, g_mask), num_seg


    def visualize(self, mask_vectors):
        color_template = {
            '0': [196, 182, 166],  # building
            '1': [0, 181, 226],  # sky
            '2': [0, 110, 51],  # tree
            '3': [196, 211, 0],  # grass
            '4': [237, 29, 36],  # person
            '5': [185, 71, 0],  # earth, ground
            '6': [255, 188, 217],  # mountain, mount
            '7': [47, 249, 36],  # plant
            '8': [172, 55, 238],  # water
            '9': [110, 98, 89],  # house
            '10': [0, 68, 129],  # sea
        }
        image = np.zeros([256, 256, 3], dtype=np.uint8)
        for i in range(len(self.target_class)):
            boolMask = mask_vectors[i] > 0.5
            image[boolMask.cpu()] = color_template[str(i)]

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr