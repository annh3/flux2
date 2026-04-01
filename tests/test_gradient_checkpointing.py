"""
Unit tests for gradient checkpointing in Flux2.

Uses a tiny model with random inputs — no pretrained weights needed.
Profiles peak CUDA memory and wall-clock time for the forward+backward pass,
comparing gradient_checkpointing=False vs gradient_checkpointing=True.
"""

import time
import tracemalloc
from dataclasses import dataclass, field

import torch
import pytest

from flux2.model import Flux2


# ---------------------------------------------------------------------------
# Tiny model params that keep tests fast on CPU
# ---------------------------------------------------------------------------

@dataclass
class TinyParams:
    in_channels: int = 16
    context_in_dim: int = 32
    hidden_size: int = 128   # hidden_size / num_heads = 128 = sum(axes_dim)
    num_heads: int = 1
    depth: int = 1
    depth_single_blocks: int = 2
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


# ---------------------------------------------------------------------------
# Helpers to build position ID tensors
# ---------------------------------------------------------------------------

def make_img_ids(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """[B, h*w, 4] integer IDs matching the format produced by prc_img."""
    ids = torch.cartesian_prod(
        torch.arange(1),          # t
        torch.arange(h),          # h
        torch.arange(w),          # w
        torch.arange(1),          # l
    )                             # [h*w, 4]
    return ids.unsqueeze(0).expand(batch, -1, -1).to(device)


def make_txt_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """[B, seq_len, 4] integer IDs matching the format produced by prc_txt."""
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(seq_len),
    )                             # [seq_len, 4]
    return ids.unsqueeze(0).expand(batch, -1, -1).to(device)


# ---------------------------------------------------------------------------
# Forward + backward runner
# ---------------------------------------------------------------------------

def run_forward_backward(model: Flux2, device: torch.device, dtype: torch.dtype):
    """Single forward + backward pass with fixed-seed random inputs."""
    B, H, W, T = 1, 4, 4, 8
    L = H * W

    torch.manual_seed(0)
    x        = torch.randn(B, L, model.in_channels,       device=device, dtype=dtype, requires_grad=True)
    x_ids    = make_img_ids(B, H, W, device)
    t        = torch.full((B,), 0.5, device=device, dtype=dtype)
    ctx      = torch.randn(B, T, model.txt_in.weight.shape[1], device=device, dtype=dtype)
    ctx_ids  = make_txt_ids(B, T, device)

    out = model(x=x, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids, guidance=None)
    loss = out.sum()
    loss.backward()
    return out.detach()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGradientCheckpointing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch.bfloat16
        self.params = TinyParams()

    def _make_model(self, gradient_checkpointing: bool) -> Flux2:
        model = Flux2(self.params, gradient_checkpointing=gradient_checkpointing)
        return model.to(device=self.device, dtype=self.dtype).train()

    def test_flag_stored_on_init(self):
        assert Flux2(self.params, gradient_checkpointing=False).gradient_checkpointing is False
        assert Flux2(self.params, gradient_checkpointing=True).gradient_checkpointing  is True

    def test_outputs_match(self):
        """GC and non-GC models with identical weights should produce identical outputs."""
        model_plain = self._make_model(gradient_checkpointing=False)
        model_gc    = self._make_model(gradient_checkpointing=True)
        model_gc.load_state_dict(model_plain.state_dict())

        out_plain = run_forward_backward(model_plain, self.device, self.dtype)
        out_gc    = run_forward_backward(model_gc,    self.device, self.dtype)

        assert torch.allclose(out_plain, out_gc, atol=1e-3), (
            f"Max abs diff: {(out_plain - out_gc).abs().max().item():.6f}"
        )

    def test_gradients_exist(self):
        """Both variants should produce non-None gradients after backward."""
        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype)
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[gc={gc}] {name} has no gradient"

    def test_memory_profiling(self, capsys):
        """
        Profile and print peak memory for plain vs gradient-checkpointed forward+backward.
        On CUDA: asserts GC peak memory <= plain peak memory.
        On CPU:  uses tracemalloc and reports allocations (no strict assertion).
        """
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
                run_forward_backward(model, self.device, self.dtype)
                torch.cuda.synchronize(self.device)
                peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            else:
                tracemalloc.start()
                run_forward_backward(model, self.device, self.dtype)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak_bytes / 1024 ** 2

            results[gc] = peak_mb

        label = "CUDA" if self.device.type == "cuda" else "CPU (tracemalloc)"
        with capsys.disabled():
            print(f"\n[memory | {label}]")
            print(f"  plain  (gc=False): {results[False]:.1f} MB")
            print(f"  checkp (gc=True):  {results[True]:.1f} MB")
            if self.device.type == "cuda":
                reduction = (results[False] - results[True]) / max(results[False], 1e-6) * 100
                print(f"  reduction: {reduction:.1f}%")

        if self.device.type == "cuda":
            assert results[True] <= results[False], (
                f"Expected GC to use <= memory, got {results[True]:.1f} MB vs {results[False]:.1f} MB"
            )

    def test_time_profiling(self, capsys):
        """
        Profile and print wall-clock time for plain vs gradient-checkpointed forward+backward.
        No strict assertion — GC trades memory for compute time.
        """
        REPEATS = 5
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            # warm-up
            run_forward_backward(model, self.device, self.dtype)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_forward_backward(model, self.device, self.dtype)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - t0) / REPEATS * 1000  # ms per iter

            results[gc] = elapsed

        with capsys.disabled():
            print(f"\n[time | {'CUDA' if self.device.type == 'cuda' else 'CPU'}]")
            print(f"  plain  (gc=False): {results[False]:.1f} ms/iter")
            print(f"  checkp (gc=True):  {results[True]:.1f} ms/iter")
            overhead = (results[True] - results[False]) / max(results[False], 1e-6) * 100
            print(f"  overhead: {overhead:+.1f}%")
