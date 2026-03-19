"""
LoRA training script for Klein9B (FLUX.2) using a HuggingFace dataset.

Usage:
    python -m flux2.training_script \
        --dataset gigant/oldbookillustrations \
        --caption_field title \
        --output_dir lora_output

The DataLoader returns raw PIL images and caption strings.
Encoding (VAE + Qwen3) happens on GPU inside the training loop,
not inside DataLoader workers (the encoders are too large to live there).

Because images have variable sizes, batch_size=1 is the default.
For larger batches, add a bucketed sampler that groups images by resolution.

#### Training Script Launch ####
python -m flux2.training_script \
    --device mps \
    --steps 2 \
    --val_every 2 \
    --log_every 1 \
    --save_every 9999 \
    --val_size 2 \
    --val_num_steps 5 \
    --output_dir lora_output

#### Tenssorboard Launch ####
(local) tensorboard --logdir lora_output/runs 

remote launch
(remote) tensorboard --logdir lora_output/runs --host 0.0.0.0 --port 6006
(local) ssh -L 6006:localhost:6006 user@remote-host
(local) http://localhost:6006
"""

import argparse
import os

import einops
import torch
import torch.nn as nn
import torchvision
from datasets import load_dataset
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from flux2.sampling import batched_prc_img, batched_prc_txt, default_prep, denoise, get_schedule, prc_img, prc_txt
from flux2.util import load_ae, load_flow_model, load_text_encoder


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter A·B."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = linear
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def _replace(module: nn.Module, name: str, rank: int, alpha: float):
    linear = getattr(module, name)
    assert isinstance(linear, nn.Linear), f"{name} is not nn.Linear"
    setattr(module, name, LoRALinear(linear, rank=rank, alpha=alpha))


def inject_lora(model, rank: int = 16, alpha: float = 16.0) -> None:
    """
    Inject LoRA into:
      - DoubleStreamBlock ×8: img_attn.qkv, img_attn.proj, txt_attn.qkv, txt_attn.proj
      - SingleStreamBlock ×24: linear1, linear2
    """
    for block in model.double_blocks:
        _replace(block.img_attn, "qkv", rank, alpha)
        _replace(block.img_attn, "proj", rank, alpha)
        _replace(block.txt_attn, "qkv", rank, alpha)
        _replace(block.txt_attn, "proj", rank, alpha)

    for block in model.single_blocks:
        _replace(block, "linear1", rank, alpha)
        _replace(block, "linear2", rank, alpha)

    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)


def lora_state_dict(model) -> dict[str, Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora(model, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    if bad := [k for k in missing if "lora" in k]:
        raise RuntimeError(f"Missing LoRA keys: {bad}")


# ---------------------------------------------------------------------------
# Timestep sampling  (logit-normal, standard for rectified flow)
# ---------------------------------------------------------------------------

def sample_timesteps(batch_size: int, device: torch.device, mean: float = 0.0, std: float = 1.0) -> Tensor:
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


# ---------------------------------------------------------------------------
# HuggingFace dataset wrapper
# ---------------------------------------------------------------------------

class HFImageDataset(Dataset):
    """
    Wraps a HuggingFace (Arrow) dataset split. Returns raw PIL images and caption strings;
    encoding happens in encode_batch() inside the training loop.
    Pass a pre-split HF dataset object (not a name) so train/val are non-overlapping.
    """

    def __init__(self, hf_split, caption_field: str, image_field: str):
        self.ds = hf_split
        self.caption_field = caption_field
        self.image_field = image_field

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        image: Image.Image = row[self.image_field]
        caption: str = row.get(self.caption_field, "") or ""
        return {"image": image, "caption": caption}


def collate_pil(batch: list[dict]) -> dict:
    """Collate a batch of dicts containing PIL images and strings."""
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }


# ---------------------------------------------------------------------------
# On-GPU encoding  (runs in the training loop, not in DataLoader workers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(
    ae,
    text_encoder,
    images: list[Image.Image],
    captions: list[str],
    device: torch.device,
    dtype: torch.dtype,
    limit_pixels: int = 1024 ** 2,
) -> dict:
    """
    Encode a batch of PIL images + caption strings into the tensors the
    model expects: img_latents, img_ids, txt_embeds, txt_ids.

    Returns a dict with all four tensors on `device`.
    """
    # --- Image encoding ---
    # default_prep: RGB, cap pixels, center-crop to multiple of 16, -> tensor [-1, 1]
    img_tensors = [
        default_prep(img, limit_pixels=limit_pixels).to(device=device, dtype=dtype)
        for img in images
    ]

    # VAE encode: each [3, H, W] -> [128, H//8, W//8]  (32 ch × 2×2 patch)
    img_latents_list = [ae.encode(t[None])[0] for t in img_tensors]   # list of [128, h, w]

    # prc_img: [128, h, w] -> (tokens [h*w, 128], ids [h*w, 4])
    img_tokens, img_ids = batched_prc_img(img_latents_list)  # [B, L, 128], [B, L, 4]

    img_tokens = img_tokens.to(device=device, dtype=dtype)
    img_ids = img_ids.to(device=device)

    # --- Text encoding ---
    txt_embeds = text_encoder(captions)                        # [B, T, context_in_dim]
    txt_tokens, txt_ids = batched_prc_txt(txt_embeds)          # same [B, T, D], [B, T, 4]

    txt_tokens = txt_tokens.to(device=device, dtype=dtype)
    txt_ids = txt_ids.to(device=device)

    return {
        "img_latents": img_tokens,
        "img_ids": img_ids,
        "txt_embeds": txt_tokens,
        "txt_ids": txt_ids,
    }


VALIDATION_PROMPT = "A bald man with a long beard smiles faintly as he carries on his back a little girl wearing a hat"


# ---------------------------------------------------------------------------
# Validation: image generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_generate(
    model,
    ae,
    text_encoder,
    step: int,
    output_dir: str,
    writer: SummaryWriter,
    device: torch.device,
    dtype: torch.dtype,
    height: int = 512,
    width: int = 512,
    num_steps: int = 20,
) -> None:
    """Run a full denoising pass with VALIDATION_PROMPT, save the image, and log to TensorBoard."""
    model.eval()

    txt_embeds = text_encoder([VALIDATION_PROMPT])              # [1, T, 12288]
    txt_tokens, txt_ids = prc_txt(txt_embeds[0])                # [T, 12288], [T, 4]
    txt_tokens = txt_tokens.unsqueeze(0).to(device=device, dtype=dtype)
    txt_ids    = txt_ids.unsqueeze(0).to(device=device)

    # Latent spatial dims: VAE downsamples 8×, then 2×2 patch → divide by 16
    h_lat = height // 16
    w_lat = width  // 16
    noise_latent = torch.randn(128, h_lat, w_lat, device=device, dtype=dtype)

    img_tokens, img_ids = prc_img(noise_latent)                 # [h_lat*w_lat, 128], [h_lat*w_lat, 4]
    img_tokens = img_tokens.unsqueeze(0)
    img_ids    = img_ids.unsqueeze(0).to(device=device)

    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # guidance is ignored by Klein9B (use_guidance_embed=False)
    denoised = denoise(
        model,
        img=img_tokens,
        img_ids=img_ids,
        txt=txt_tokens,
        txt_ids=txt_ids,
        timesteps=timesteps,
        guidance=1.0,
    )                                                           # [1, L, 128]

    denoised_spatial = einops.rearrange(denoised[0], "(h w) c -> c h w", h=h_lat, w=w_lat)
    pixel = ae.decode(denoised_spatial.unsqueeze(0))            # [1, 3, H, W] in [-1, 1]
    pixel = (pixel.clamp(-1, 1) + 1) / 2                       # [0, 1]

    img_pil = torchvision.transforms.ToPILImage()(pixel[0].float().cpu())
    val_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    img_pil.save(os.path.join(val_dir, f"step_{step:06d}.png"))

    writer.add_image("validation/generated", pixel[0].float().cpu(), global_step=step)
    print(f"[val] image saved → {val_dir}/step_{step:06d}.png")

    model.train()


# ---------------------------------------------------------------------------
# Validation: loss on held-out set
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_loss(
    model,
    ae,
    text_encoder,
    val_items: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    limit_pixels: int,
) -> float:
    model.eval()
    total_loss = 0.0

    for item in val_items:
        encoded = encode_batch(
            ae, text_encoder,
            images=[item["image"]],
            captions=[item["caption"]],
            device=device,
            dtype=dtype,
            limit_pixels=limit_pixels,
        )
        img_latents = encoded["img_latents"]
        img_ids     = encoded["img_ids"]
        txt_embeds  = encoded["txt_embeds"]
        txt_ids     = encoded["txt_ids"]

        noise = torch.randn_like(img_latents)
        t = sample_timesteps(1, device)
        t_exp = t[:, None, None]
        x_t = t_exp * img_latents + (1.0 - t_exp) * noise
        target_velocity = img_latents - noise

        pred = model(x=x_t, x_ids=img_ids, timesteps=t, ctx=txt_embeds, ctx_ids=txt_ids, guidance=None)
        total_loss += torch.mean((pred - target_velocity) ** 2).item()

    model.train()
    return total_loss / len(val_items)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # --- Load model, VAE, text encoder via util.py helpers ---
    model = load_flow_model("flux.2-klein-9b", device=device)
    ae = load_ae("flux.2-klein-9b", device=device)
    text_encoder = load_text_encoder("flux.2-klein-9b", device=device)

    ae.eval()
    text_encoder.eval()

    inject_lora(model, rank=args.rank, alpha=args.alpha)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # --- Data ---
    raw = load_dataset(args.dataset, split=args.split)
    splits = raw.train_test_split(test_size=args.val_size, seed=42)
    train_split = HFImageDataset(splits["train"], caption_field=args.caption_field, image_field=args.image_field)
    val_items   = [HFImageDataset(splits["test"],  caption_field=args.caption_field, image_field=args.image_field)[i]
                   for i in range(len(splits["test"]))]
    print(f"Train: {len(train_split)} | Val: {len(val_items)} examples")

    loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    model.train()
    step = 0
    loss_accum = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            # Encode on GPU (VAE + text encoder), no grad needed
            encoded = encode_batch(
                ae, text_encoder,
                images=batch["images"],
                captions=batch["captions"],
                device=device,
                dtype=dtype,
                limit_pixels=args.limit_pixels,
            )

            img_latents = encoded["img_latents"]   # [B, L, 128]
            img_ids     = encoded["img_ids"]        # [B, L, 4]
            txt_embeds  = encoded["txt_embeds"]     # [B, T, 12288]
            txt_ids     = encoded["txt_ids"]        # [B, T, 4]
            B = img_latents.shape[0]

            # --- Rectified flow forward process ---
            noise = torch.randn_like(img_latents)
            t = sample_timesteps(B, device)                    # [B]
            t_exp = t[:, None, None]                           # [B, 1, 1]
            x_t = t_exp * img_latents + (1.0 - t_exp) * noise
            target_velocity = img_latents - noise

            # --- Forward pass (Klein9B: guidance=None) ---
            pred_velocity = model(
                x=x_t,
                x_ids=img_ids,
                timesteps=t,
                ctx=txt_embeds,
                ctx_ids=txt_ids,
                guidance=None,
            )

            loss = torch.mean((pred_velocity - target_velocity) ** 2)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            loss_accum += loss.item()

            if step % args.log_every == 0:
                avg_loss = loss_accum / args.log_every
                print(f"step {step:6d} | train_loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                writer.add_scalar("loss/train", avg_loss, global_step=step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=step)
                loss_accum = 0.0

            if step % args.val_every == 0:
                val_loss = validate_loss(model, ae, text_encoder, val_items, device, dtype, args.limit_pixels)
                print(f"step {step:6d} | val_loss   {val_loss:.4f}")
                writer.add_scalar("loss/val", val_loss, global_step=step)
                validate_generate(
                    model, ae, text_encoder,
                    step=step,
                    output_dir=args.output_dir,
                    writer=writer,
                    device=device,
                    dtype=dtype,
                    height=args.val_height,
                    width=args.val_width,
                    num_steps=args.val_num_steps,
                )

            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"lora_step{step:06d}.pt")
                torch.save(lora_state_dict(model), ckpt)
                print(f"Saved: {ckpt}")

    final = os.path.join(args.output_dir, "lora_final.pt")
    torch.save(lora_state_dict(model), final)
    print(f"Done. LoRA saved to {final}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="gigant/oldbookillustrations")
    p.add_argument("--split",         default="train")
    p.add_argument("--caption_field", default="image_description", help="HF dataset column to use as prompt")
    p.add_argument("--image_field",   default="1600px",            help="HF dataset column containing PIL images")
    p.add_argument("--output_dir",    default="lora_output")
    p.add_argument("--rank",          type=int,   default=16)
    p.add_argument("--alpha",         type=float, default=16.0)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=1,
                   help="batch_size>1 requires images of identical resolution or a bucketed sampler")
    p.add_argument("--steps",         type=int,   default=1000)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--save_every",    type=int,   default=500)
    p.add_argument("--num_workers",   type=int,   default=0,
                   help="keep 0: PIL images can't be pickled reliably by DataLoader workers")
    p.add_argument("--limit_pixels",  type=int,   default=1024**2)
    # Validation
    p.add_argument("--val_every",     type=int,   default=250)
    p.add_argument("--val_size",      type=int,   default=50,  help="number of held-out examples for val loss")
    p.add_argument("--val_height",    type=int,   default=512)
    p.add_argument("--val_width",     type=int,   default=512)
    p.add_argument("--val_num_steps", type=int,   default=20,  help="denoising steps for validation image")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
