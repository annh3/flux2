"""
Compare full Klein9B vs. a pruned variant (blocks dropped) on a fixed set of prompts.

Usage:
    # Manual: specify block indices to drop
    python -m flux2.compare_pruned \
        --drop_double 3 7 \
        --drop_single 20 23 \
        --output_dir artifacts

    # Automatic: drop the top-N most important blocks (highest MAD)
    python -m flux2.compare_pruned \
        --drop_n 2 \
        --importance_json artifacts/block_importance.json \
        --output_dir artifacts
"""

import argparse
import json
import os

import torch
from PIL import Image

from flux2.sampling import batched_prc_img, denoise, get_schedule, prc_txt, scatter_ids
from flux2.util import load_ae, load_flow_model, load_text_encoder

PROMPTS = [
    "A young man has sat down by a tree to have something to eat, of which he gives a morsel to his dog",
    "A young man is drinking from a jug under the gaze of an older man filling another one from a keg",
    "An old man with a long beard and tattered clothes wakes up outdoors and checks his head and his gun",
]

HEIGHT = 768
WIDTH = 768
NUM_STEPS = 4


def prune_model(model, drop_double, drop_single):
    import torch.nn as nn

    if drop_double:
        keep = [b for i, b in enumerate(model.double_blocks) if i not in drop_double]
        model.double_blocks = nn.ModuleList(keep)
    if drop_single:
        keep = [b for i, b in enumerate(model.single_blocks) if i not in drop_single]
        model.single_blocks = nn.ModuleList(keep)
    return model


@torch.no_grad()
def generate_images(model, ae, text_encoder, prompts, device, dtype, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    h, w = HEIGHT // 16, WIDTH // 16

    for idx, prompt in enumerate(prompts):
        noise = torch.randn(1, 128, h, w, device=device, dtype=dtype)
        img_tok, img_ids = batched_prc_img(noise)
        img_tok = img_tok.to(device=device, dtype=dtype)
        img_ids = img_ids.to(device=device)

        txt_embed = text_encoder([prompt])
        txt_tok, txt_ids = prc_txt(txt_embed[0])
        txt_tok = txt_tok.unsqueeze(0).to(device=device, dtype=dtype)
        txt_ids = txt_ids.unsqueeze(0).to(device=device)

        timesteps = get_schedule(NUM_STEPS, img_tok.shape[1])

        x = denoise(model, img_tok, img_ids, txt_tok, txt_ids, timesteps, guidance=1.0)

        x_spatial = torch.cat(scatter_ids(x, img_ids)).squeeze(2)
        img_decoded = ae.decode(x_spatial)
        img_decoded = img_decoded.clamp(-1, 1)

        arr = (127.5 * (img_decoded[0].permute(1, 2, 0) + 1.0)).cpu().byte().numpy()
        pil = Image.fromarray(arr)
        fname = os.path.join(output_dir, f"prompt_{idx + 1:02d}.png")
        pil.save(fname)
        print(f"  Saved {fname}")


def load_drop_indices(importance_json, drop_n):
    with open(importance_json) as f:
        data = json.load(f)
    ranked = data["ranked"]  # sorted descending by MAD
    top_n = ranked[:drop_n]
    drop_double = {e["index"] for e in top_n if e["type"] == "double"}
    drop_single = {e["index"] for e in top_n if e["type"] == "single"}
    print(f"Top-{drop_n} most important blocks to drop:")
    for e in top_n:
        print(f"  {e['type']:>6} block {e['index']:>2}  MAD={e['mad']:.6f}")
    return drop_double, drop_single


def main(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("Loading models...")
    ae = load_ae("flux.2-klein-9b", device=device)
    text_encoder = load_text_encoder("flux.2-klein-9b", device=device)
    ae.eval()
    text_encoder.eval()

    if args.drop_n is not None:
        assert args.importance_json, "--importance_json is required when using --drop_n"
        drop_double, drop_single = load_drop_indices(args.importance_json, args.drop_n)
    else:
        drop_double = set(args.drop_double or [])
        drop_single = set(args.drop_single or [])

    # --- Full model ---
    print("Generating with full model...")
    model_full = load_flow_model("flux.2-klein-9b", device=device)
    model_full.eval()
    full_dir = os.path.join(args.output_dir, "FLUX.2_klein")
    generate_images(model_full, ae, text_encoder, PROMPTS, device, dtype, full_dir)
    del model_full
    torch.cuda.empty_cache()

    # --- Pruned model ---
    print(f"Generating with pruned model (drop_double={sorted(drop_double)}, drop_single={sorted(drop_single)})...")
    model_pruned = load_flow_model("flux.2-klein-9b", device=device)
    model_pruned = prune_model(model_pruned, drop_double, drop_single)
    model_pruned.eval()
    pruned_dir = os.path.join(args.output_dir, "FLUX.2_klein_pruned")
    generate_images(model_pruned, ae, text_encoder, PROMPTS, device, dtype, pruned_dir)
    del model_pruned

    print(f"\nDone. Full model images: {full_dir}")
    print(f"Pruned model images:     {pruned_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drop_double", type=int, nargs="*", default=[], metavar="IDX",
                   help="Double-stream block indices to drop (manual)")
    p.add_argument("--drop_single", type=int, nargs="*", default=[], metavar="IDX",
                   help="Single-stream block indices to drop (manual)")
    p.add_argument("--drop_n", type=int, default=None,
                   help="Automatically drop the top-N most important blocks from block_importance.json")
    p.add_argument("--importance_json", default="artifacts/block_importance.json",
                   help="Path to block_importance.json produced by block_importance.py")
    p.add_argument("--output_dir", default="artifacts")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
