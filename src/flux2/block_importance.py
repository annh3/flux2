"""
Measure the importance of each block in Klein9B using Mean Absolute Deviation
(MAD) of residual contributions over a calibration dataset (test split).

Usage:
    python -m flux2.block_importance \
        --dataset gigant/oldbookillustrations \
        --val_size 50 \
        --output_dir artifacts
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

from flux2.model import timestep_embedding
from flux2.sampling import default_prep, prc_img, prc_txt
from flux2.util import load_ae, load_flow_model, load_text_encoder


@torch.no_grad()
def measure_importance(model, ae, text_encoder, items, device, dtype, limit_pixels=1024**2):
    """
    Replicate Flux2.forward for each calibration example, capturing the per-block
    residual (output - input) and accumulating its MAD.
    Uses logit-normal timestep sampling (same as training).
    """
    n_double = len(model.double_blocks)
    n_single = len(model.single_blocks)
    double_mads = [0.0] * n_double
    single_mads = [0.0] * n_single
    count = 0

    for item in items:
        ae_dtype = next(ae.parameters()).dtype
        try:
            img_tensor = default_prep(item["image"], limit_pixels=limit_pixels).to(device=device, dtype=ae_dtype)
            img_latent = ae.encode(img_tensor[None])[0]          # [128, h, w]
        except Exception as e:
            print(f"  Skipping example: {e}")
            continue

        img_tok, img_ids = prc_img(img_latent)
        img_tok = img_tok.unsqueeze(0).to(device=device, dtype=dtype)
        img_ids = img_ids.unsqueeze(0).to(device=device)

        txt_embed = text_encoder([item["caption"]])              # [1, T, 12288]
        txt_tok, txt_ids = prc_txt(txt_embed[0])
        txt_tok = txt_tok.unsqueeze(0).to(device=device, dtype=dtype)
        txt_ids = txt_ids.unsqueeze(0).to(device=device)
        num_txt_tokens = txt_tok.shape[1]

        # Logit-normal timestep (same distribution as training)
        t = torch.sigmoid(torch.randn(1, device=device, dtype=dtype))
        noise = torch.randn_like(img_tok)
        x_t = t[:, None, None] * noise + (1.0 - t[:, None, None]) * img_tok

        # --- Mirror Flux2.forward, capturing residuals at each block ---
        vec = model.time_in(timestep_embedding(t, 256))

        double_mod_img = model.double_stream_modulation_img(vec)
        double_mod_txt = model.double_stream_modulation_txt(vec)
        single_mod, _ = model.single_stream_modulation(vec)

        img = model.img_in(x_t)
        txt = model.txt_in(txt_tok)
        pe_x  = model.pe_embedder(img_ids)
        pe_ctx = model.pe_embedder(txt_ids)

        for i, block in enumerate(model.double_blocks):
            img_before = img.clone()
            txt_before = txt.clone()
            img, txt, _ = block.forward_kv_extract(
                img, txt, pe_x, pe_ctx,
                double_mod_img, double_mod_txt,
                num_ref_tokens=0,
            )
            residual = torch.cat([img - img_before, txt - txt_before], dim=1)
            double_mads[i] += residual.abs().mean().item()

        img = torch.cat((txt, img), dim=1)
        pe  = torch.cat((pe_ctx, pe_x), dim=2)

        for i, block in enumerate(model.single_blocks):
            x_before = img.clone()
            img, _ = block.forward_kv_extract(
                img, pe, single_mod, num_txt_tokens, num_ref_tokens=0,
            )
            single_mads[i] += (img - x_before).abs().mean().item()

        count += 1
        print(f"  [{count}/{len(items)}]", end="\r")

    print()
    double_mads = [m / count for m in double_mads]
    single_mads = [m / count for m in single_mads]
    return double_mads, single_mads


def rank_blocks(double_mads, single_mads):
    entries = []
    for i, m in enumerate(double_mads):
        entries.append({"type": "double", "index": i, "mad": m})
    for i, m in enumerate(single_mads):
        entries.append({"type": "single", "index": i, "mad": m})
    return sorted(entries, key=lambda x: x["mad"], reverse=True)


def plot_importance(double_mads, single_mads, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(range(len(double_mads)), double_mads, color="steelblue")
    ax1.set_title("Double Stream Blocks — MAD of Residual Contribution")
    ax1.set_xlabel("Block index")
    ax1.set_ylabel("MAD")
    ax1.set_xticks(range(len(double_mads)))

    ax2.bar(range(len(single_mads)), single_mads, color="darkorange")
    ax2.set_title("Single Stream Blocks — MAD of Residual Contribution")
    ax2.set_xlabel("Block index")
    ax2.set_ylabel("MAD")
    ax2.set_xticks(range(len(single_mads)))

    plt.tight_layout()
    path = os.path.join(output_dir, "block_importance.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


def main(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("Loading models...")
    model = load_flow_model("flux.2-klein-9b", device=device)
    ae = load_ae("flux.2-klein-9b", device=device)
    text_encoder = load_text_encoder("flux.2-klein-9b", device=device)
    model.eval()
    ae.eval()
    text_encoder.eval()

    print("Loading calibration data (test split)...")
    raw = load_dataset(args.dataset, split="train")
    splits = raw.train_test_split(test_size=args.val_size, seed=42)
    val_ds = splits["test"]

    items = []
    for i in range(len(val_ds)):
        try:
            row = val_ds[i]
            img = row[args.image_field]
            img.load()
            items.append({"image": img, "caption": row.get(args.caption_field, "") or ""})
        except OSError:
            continue
    print(f"Calibration set: {len(items)} examples")

    double_mads, single_mads = measure_importance(
        model, ae, text_encoder, items, device, dtype, args.limit_pixels
    )

    ranked = rank_blocks(double_mads, single_mads)

    print(f"\n{'Rank':>4}  {'Type':>6}  {'Index':>5}  {'MAD':>10}")
    print("-" * 34)
    for rank, e in enumerate(ranked, 1):
        print(f"{rank:>4}  {e['type']:>6}  {e['index']:>5}  {e['mad']:>10.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "block_importance.json")
    with open(json_path, "w") as f:
        json.dump({"double": double_mads, "single": single_mads, "ranked": ranked}, f, indent=2)
    print(f"\nScores saved to {json_path}")

    plot_importance(double_mads, single_mads, args.output_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="gigant/oldbookillustrations")
    p.add_argument("--caption_field", default="image_description")
    p.add_argument("--image_field",   default="1600px")
    p.add_argument("--val_size",      type=int, default=50)
    p.add_argument("--limit_pixels",  type=int, default=1024**2)
    p.add_argument("--output_dir",    default="artifacts")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
