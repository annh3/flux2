"""
Knowledge distillation for a pruned Klein9B model.

The teacher is the full Klein9B (frozen). The student is a pruned copy
(blocks removed). The distillation loss matches the student's velocity
prediction to the teacher's at every sampled timestep:

    L = ||v_student(x_t, t) - v_teacher(x_t, t)||²

LoRA adapters are injected into the student so that base weights stay frozen
and only the low-rank deltas are updated.

Usage:
    python -m flux2.distill_pruned \
        --importance_json artifacts/block_importance.json \
        --drop_n 2 \
        --output_dir distill_output

#### TensorBoard Launch ####
# local:
#   tensorboard --logdir distill_output/runs
#   open http://localhost:6006
#
# remote:
#   (remote) tensorboard --logdir ~/flux2/src/flux2/distill_output/runs --host 0.0.0.0 --port 6006
#   (local)  ssh -L 6006:localhost:6006 ubuntu@<server-ip>
#   (local)  open http://localhost:6006
"""

import argparse
import json
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

VALIDATION_PROMPTS = [
    "A young man has sat down by a tree to have something to eat, of which he gives a morsel to his dog",
    "A young man is drinking from a jug under the gaze of an older man filling another one from a keg",
    "An old man with a long beard and tattered clothes wakes up outdoors and checks his head and his gun",
]


# ---------------------------------------------------------------------------
# Block pruning
# ---------------------------------------------------------------------------

def prune_model(model, drop_double: set, drop_single: set):
    if drop_double:
        keep = [b for i, b in enumerate(model.double_blocks) if i not in drop_double]
        model.double_blocks = nn.ModuleList(keep)
    if drop_single:
        keep = [b for i, b in enumerate(model.single_blocks) if i not in drop_single]
        model.single_blocks = nn.ModuleList(keep)
    return model


def load_drop_indices(importance_json: str, drop_n: int):
    with open(importance_json) as f:
        data = json.load(f)
    ranked = data["ranked"]
    top_n = ranked[:drop_n]
    drop_double = {e["index"] for e in top_n if e["type"] == "double"}
    drop_single = {e["index"] for e in top_n if e["type"] == "single"}
    print(f"Dropping top-{drop_n} most important blocks:")
    for e in top_n:
        print(f"  {e['type']:>6} block {e['index']:>2}  MAD={e['mad']:.6f}")
    return drop_double, drop_single


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = linear
        self.scale = alpha / rank
        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(model, rank: int, alpha: float) -> None:
    for block in model.double_blocks:
        for attr in ("qkv", "proj"):
            for stream in (block.img_attn, block.txt_attn):
                linear = getattr(stream, attr)
                setattr(stream, attr, LoRALinear(linear, rank=rank, alpha=alpha))

    for block in model.single_blocks:
        for attr in ("linear1", "linear2"):
            linear = getattr(block, attr)
            setattr(block, attr, LoRALinear(linear, rank=rank, alpha=alpha))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.to(device=device, dtype=dtype)
            module.lora_B.to(device=device, dtype=dtype)
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)


def lora_state_dict(model) -> dict[str, Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

def sample_timesteps(batch_size: int, device: torch.device) -> Tensor:
    return torch.sigmoid(torch.randn(batch_size, device=device))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFImageDataset(Dataset):
    def __init__(self, hf_split, caption_field: str, image_field: str):
        self.ds = hf_split
        self.caption_field = caption_field
        self.image_field = image_field

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        while True:
            try:
                row = self.ds[idx]
                image: Image.Image = row[self.image_field]
                image.load()
                caption: str = row.get(self.caption_field, "") or ""
                return {"image": image, "caption": caption}
            except OSError:
                idx = (idx + 1) % len(self.ds)


def collate_pil(batch: list[dict]) -> dict:
    return {"images": [item["image"] for item in batch], "captions": [item["caption"] for item in batch]}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(ae, text_encoder, images, captions, device, dtype, limit_pixels=1024**2) -> dict:
    ae_dtype = next(ae.parameters()).dtype
    img_tensors = [default_prep(img, limit_pixels=limit_pixels).to(device=device, dtype=ae_dtype) for img in images]
    img_latents_list = [ae.encode(t[None])[0] for t in img_tensors]
    img_tokens, img_ids = batched_prc_img(img_latents_list)
    img_tokens = img_tokens.to(device=device, dtype=dtype)
    img_ids = img_ids.to(device=device)

    txt_embeds = text_encoder(captions)
    txt_tokens, txt_ids = batched_prc_txt(txt_embeds)
    txt_tokens = txt_tokens.to(device=device, dtype=dtype)
    txt_ids = txt_ids.to(device=device)

    return {"img_latents": img_tokens, "img_ids": img_ids, "txt_embeds": txt_tokens, "txt_ids": txt_ids}


# ---------------------------------------------------------------------------
# Validation: distillation loss on held-out set
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_loss(teacher, student, ae, text_encoder, val_items, device, dtype, limit_pixels) -> float:
    student.eval()
    total = 0.0
    for item in val_items:
        enc = encode_batch(ae, text_encoder, [item["image"]], [item["caption"]], device, dtype, limit_pixels)
        img_latents = enc["img_latents"]
        img_ids     = enc["img_ids"]
        txt_embeds  = enc["txt_embeds"]
        txt_ids     = enc["txt_ids"]

        noise = torch.randn_like(img_latents)
        t = sample_timesteps(1, device).to(dtype)
        x_t = t[:, None, None] * noise + (1.0 - t[:, None, None]) * img_latents

        v_teacher = teacher(x=x_t, x_ids=img_ids, timesteps=t, ctx=txt_embeds, ctx_ids=txt_ids, guidance=None)
        v_student = student(x=x_t, x_ids=img_ids, timesteps=t, ctx=txt_embeds, ctx_ids=txt_ids, guidance=None)
        total += torch.mean((v_student - v_teacher) ** 2).item()

    student.train()
    return total / len(val_items)


# ---------------------------------------------------------------------------
# Validation: image generation for each validation prompt
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_generate(student, ae, text_encoder, step, output_dir, writer, device, dtype,
                      height=512, width=512, num_steps=4) -> None:
    student.eval()
    val_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)

    h_lat, w_lat = height // 16, width // 16

    for i, prompt in enumerate(VALIDATION_PROMPTS):
        txt_embeds = text_encoder([prompt])
        txt_tokens, txt_ids = prc_txt(txt_embeds[0])
        txt_tokens = txt_tokens.unsqueeze(0).to(device=device, dtype=dtype)
        txt_ids    = txt_ids.unsqueeze(0).to(device=device)

        noise_latent = torch.randn(128, h_lat, w_lat, device=device, dtype=dtype)
        img_tokens, img_ids = prc_img(noise_latent)
        img_tokens = img_tokens.unsqueeze(0)
        img_ids    = img_ids.unsqueeze(0).to(device=device)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])
        denoised = denoise(student, img=img_tokens, img_ids=img_ids, txt=txt_tokens, txt_ids=txt_ids,
                           timesteps=timesteps, guidance=1.0)

        denoised_spatial = einops.rearrange(denoised[0], "(h w) c -> c h w", h=h_lat, w=w_lat)
        pixel = ae.decode(denoised_spatial.unsqueeze(0))
        pixel = (pixel.clamp(-1, 1) + 1) / 2

        img_pil = torchvision.transforms.ToPILImage()(pixel[0].float().cpu())
        fname = os.path.join(val_dir, f"step_{step:06d}_prompt{i + 1}.png")
        img_pil.save(fname)
        writer.add_image(f"validation/prompt_{i + 1}", pixel[0].float().cpu(), global_step=step)
        print(f"[val] saved {fname}")

    student.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Shared encoders
    ae = load_ae("flux.2-klein-9b", device=device)
    text_encoder = load_text_encoder("flux.2-klein-9b", device=device)
    ae.eval()
    text_encoder.eval()

    # Teacher (full model, frozen)
    print("Loading teacher (full model)...")
    teacher = load_flow_model("flux.2-klein-9b", device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student (pruned model + LoRA)
    print("Loading student (pruned model)...")
    drop_double, drop_single = load_drop_indices(args.importance_json, args.drop_n)
    student = load_flow_model("flux.2-klein-9b", device=device)
    student = prune_model(student, drop_double, drop_single)
    inject_lora(student, rank=args.rank, alpha=args.alpha)

    trainable = [p for p in student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"Student trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # Dataset
    raw = load_dataset(args.dataset, split=args.split)
    splits = raw.train_test_split(test_size=args.val_size, seed=42)
    train_ds = HFImageDataset(splits["train"], caption_field=args.caption_field, image_field=args.image_field)
    val_items = [HFImageDataset(splits["test"], caption_field=args.caption_field, image_field=args.image_field)[i]
                 for i in range(len(splits["test"]))]
    print(f"Train: {len(train_ds)} | Val: {len(val_items)}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_pil)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    student.train()
    step = 0
    loss_accum = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            enc = encode_batch(ae, text_encoder, batch["images"], batch["captions"],
                               device, dtype, args.limit_pixels)
            img_latents = enc["img_latents"]
            img_ids     = enc["img_ids"]
            txt_embeds  = enc["txt_embeds"]
            txt_ids     = enc["txt_ids"]
            B = img_latents.shape[0]

            noise = torch.randn_like(img_latents)
            t = sample_timesteps(B, device).to(dtype)
            x_t = t[:, None, None] * noise + (1.0 - t[:, None, None]) * img_latents

            # Teacher prediction (no grad)
            with torch.no_grad():
                v_teacher = teacher(x=x_t, x_ids=img_ids, timesteps=t, ctx=txt_embeds, ctx_ids=txt_ids, guidance=None)

            # Student prediction
            v_student = student(x=x_t, x_ids=img_ids, timesteps=t, ctx=txt_embeds, ctx_ids=txt_ids, guidance=None)

            loss = torch.mean((v_student - v_teacher) ** 2)

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
                print(f"step {step:6d} | distill_loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                writer.add_scalar("loss/train", avg_loss, global_step=step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=step)
                loss_accum = 0.0

            if step % args.val_every == 0:
                val_loss = validate_loss(teacher, student, ae, text_encoder, val_items, device, dtype, args.limit_pixels)
                print(f"step {step:6d} | val_loss     {val_loss:.4f}")
                writer.add_scalar("loss/val", val_loss, global_step=step)
                validate_generate(student, ae, text_encoder, step=step, output_dir=args.output_dir,
                                  writer=writer, device=device, dtype=dtype,
                                  height=args.val_height, width=args.val_width, num_steps=args.val_num_steps)

            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"distill_step{step:06d}.pt")
                torch.save(lora_state_dict(student), ckpt)
                print(f"Saved: {ckpt}")

    final = os.path.join(args.output_dir, "distill_final.pt")
    torch.save(lora_state_dict(student), final)
    print(f"Done. LoRA saved to {final}")
    writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Pruning
    p.add_argument("--importance_json", default="artifacts/block_importance.json")
    p.add_argument("--drop_n",          type=int, default=2,
                   help="Number of top-MAD blocks to drop from student")
    # Dataset
    p.add_argument("--dataset",         default="gigant/oldbookillustrations")
    p.add_argument("--split",           default="train")
    p.add_argument("--caption_field",   default="image_description")
    p.add_argument("--image_field",     default="1600px")
    # LoRA
    p.add_argument("--rank",            type=int,   default=32)
    p.add_argument("--alpha",           type=float, default=32.0)
    # Optimiser
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--steps",           type=int,   default=5000)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--limit_pixels",    type=int,   default=1024**2)
    # Logging / checkpointing
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--val_every",       type=int,   default=500)
    p.add_argument("--save_every",      type=int,   default=1000)
    p.add_argument("--output_dir",      default="distill_output")
    # Validation generation
    p.add_argument("--val_size",        type=int,   default=50)
    p.add_argument("--val_height",      type=int,   default=512)
    p.add_argument("--val_width",       type=int,   default=512)
    p.add_argument("--val_num_steps",   type=int,   default=4)
    p.add_argument("--device",          default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
