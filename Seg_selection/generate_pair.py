import sys
sys.path.append('/home/yxjiang/Scenimefy-dev')

import argparse

import torch
from torchvision import utils
from Pseudo_generation.model import Generator
from tqdm import tqdm
import time
from pathlib import Path

from segmentation_net import SEMSEG

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate pseudo paired samples filtered by Mask2Former")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="truncation ratio"
    )

    parser.add_argument(
        "--ckpt1",
        type=str,
        default="../Pseudo_generation/checkpoints/lhq-220000.pt",
        help="path to the original model checkpoint",
    )

    parser.add_argument(
        "--ckpt2",
        type=str,
        default="../Pseudo_generation/checkpoints/shinkai-221000.pt",
        help="path to the finetuned model checkpoint",
    )

    parser.add_argument(
        "--num_sample",
        type=int,
        default=30,
        help="number of paired samples to be generated",
    )

    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./data/s2a_shinkai", 
        help="path to save the paired sample images"
    )

    parser.add_argument(
        "--seg_loss_th", 
        type=float, 
        default=5.0, 
        help="threshold of segmentation loss for semantic consistency"
    )

    parser.add_argument(
        "--seg_cat_th", 
        type=int,
         default=1, 
         help="threshold of detected category for semantic abundance"
    )
    
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema1 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint1 = torch.load(args.ckpt1, map_location="cpu")

    g_ema1.load_state_dict(checkpoint1["g_ema"], strict=False)

    g_ema2 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint2 = torch.load(args.ckpt2, map_location="cpu")

    g_ema2.load_state_dict(checkpoint2["g_ema"], strict=False)

    # load filter
    semseg_predictor = SEMSEG()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema1.mean_latent(4096)
    else:
        mean_latent = None

    # create output data folder
    Path(f"{args.output_path}/trainA").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_path}/trainB").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        g_ema1.eval()
        g_ema2.eval()

        num_remain = args.num_sample 
        while num_remain > 0:

            latent = g_ema1.get_latent(torch.randn(1, args.latent, device=device))

            sample1, _ = g_ema1(
                [latent], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )

            sample2, _ = g_ema2(
                [latent], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )

            seg_loss, seg_cnt = semseg_predictor.forward_ce(sample2, sample1)

            if seg_loss.item() < args.seg_loss_th and seg_cnt > args.seg_cat_th:

                utils.save_image(
                    sample1,
                    f"{args.output_path}/trainA/{num_remain-1}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    sample2,
                    f"{args.output_path}/trainB/{num_remain-1}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

                num_remain = num_remain - 1
