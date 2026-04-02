"""
Unit tests for gradient checkpointing in Flux2.

Uses randomly-initialised weights (no pretrained download needed).

Three test suites:
  - TestKlein1BMPSGradientCheckpointing: ~1B model on MPS (local Mac, Apple Silicon)
  - TestKlein9BMPSGradientCheckpointing: Klein9B model on MPS (local Mac, requires ~36 GB unified memory)
  - TestKlein9BGradientCheckpointing: Klein9B model on CUDA (GH200)

Run all tests:
    pytest tests/test_gradient_checkpointing.py -v -s

Run only the ~1B MPS tests (local Mac, Apple Silicon):
    pytest tests/test_gradient_checkpointing.py::TestKlein1BMPSGradientCheckpointing -v -s

Run only the Klein9B MPS tests (local Mac, ~36 GB unified memory required):
    pytest tests/test_gradient_checkpointing.py::TestKlein9BMPSGradientCheckpointing -v -s

Run only the Klein9B CUDA tests (requires CUDA):
    pytest tests/test_gradient_checkpointing.py::TestKlein9BGradientCheckpointing -v -s
"""

import time

import torch
import pytest

from flux2.model import Flux2, Klein1BParams, Klein9BParams


# ---------------------------------------------------------------------------
# Helpers to build position ID tensors
# ---------------------------------------------------------------------------

def make_img_ids(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """[B, h*w, 4] integer IDs matching the format produced by prc_img."""
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(h),
        torch.arange(w),
        torch.arange(1),
    )  # [h*w, 4]
    return ids.unsqueeze(0).expand(batch, -1, -1).to(device)


def make_txt_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """[B, seq_len, 4] integer IDs matching the format produced by prc_txt."""
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(seq_len),
    )  # [seq_len, 4]
    return ids.unsqueeze(0).expand(batch, -1, -1).to(device)


# ---------------------------------------------------------------------------
# Forward + backward runner
# ---------------------------------------------------------------------------

def run_forward_backward(
    model: Flux2, device: torch.device, dtype: torch.dtype,
    h: int = 32, w: int = 32, t: int = 128,
):
    """Single forward + backward pass with fixed-seed random inputs."""
    B = 1
    L = h * w

    torch.manual_seed(0)
    x       = torch.randn(B, L, model.in_channels,            device=device, dtype=dtype, requires_grad=True)
    x_ids   = make_img_ids(B, h, w, device)
    ts      = torch.full((B,), 0.5,                           device=device, dtype=dtype)
    ctx     = torch.randn(B, t, model.txt_in.weight.shape[1], device=device, dtype=dtype)
    ctx_ids = make_txt_ids(B, t, device)

    out = model(x=x, x_ids=x_ids, timesteps=ts, ctx=ctx, ctx_ids=ctx_ids, guidance=None)
    loss = out.sum()
    loss.backward()
    return out.detach()


# ---------------------------------------------------------------------------
# Tests — Klein1B model (MPS, local Mac)
# 256x256 image → 16x16 latent grid (L=256), 64 text tokens
# ---------------------------------------------------------------------------

class TestKlein1BMPSGradientCheckpointing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("mps")
        self.dtype  = torch.float32  # bfloat16 not supported on MPS
        self.params = Klein1BParams()

    def _make_model(self, gradient_checkpointing: bool) -> Flux2:
        model = Flux2(self.params, gradient_checkpointing=gradient_checkpointing)
        return model.to(device=self.device, dtype=self.dtype).train()

    def test_flag_stored_on_init(self):
        params = Klein1BParams()
        assert Flux2(params, gradient_checkpointing=False).gradient_checkpointing is False
        assert Flux2(params, gradient_checkpointing=True).gradient_checkpointing  is True

    def test_outputs_match(self):
        """GC and non-GC models with identical weights should produce identical outputs."""
        model_plain = self._make_model(gradient_checkpointing=False)
        model_gc    = self._make_model(gradient_checkpointing=True)
        model_gc.load_state_dict(model_plain.state_dict())

        out_plain = run_forward_backward(model_plain, self.device, self.dtype, h=16, w=16, t=64)
        out_gc    = run_forward_backward(model_gc,    self.device, self.dtype, h=16, w=16, t=64)

        assert torch.allclose(out_plain, out_gc, atol=1e-3), (
            f"Max abs diff: {(out_plain - out_gc).abs().max().item():.6f}"
        )

    def test_gradients_exist(self):
        """Both variants should produce non-None gradients after backward."""
        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[gc={gc}] {name} has no gradient"

    def test_memory_profiling(self, capsys):
        """
        Profile and print MPS memory for plain vs gradient-checkpointed forward+backward.
        Uses current_allocated_memory() before/after — torch.mps has no peak memory API,
        so this measures net allocation delta rather than true peak. No assertion.
        """
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            torch.mps.empty_cache()
            before_mb = torch.mps.current_allocated_memory() / 1024 ** 2
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            torch.mps.synchronize()
            after_mb = torch.mps.current_allocated_memory() / 1024 ** 2
            results[gc] = after_mb - before_mb

        with capsys.disabled():
            print(f"\n[memory | MPS (net delta) | Klein1B]")
            print(f"  plain  (gc=False): {results[False]:.1f} MB")
            print(f"  checkp (gc=True):  {results[True]:.1f} MB")
            print(f"  note: torch.mps has no peak-memory API; run CUDA test for true peak")

    def test_time_profiling(self, capsys):
        """
        Profile and print wall-clock time for plain vs gradient-checkpointed forward+backward.
        No strict assertion — GC trades memory for compute time.
        """
        REPEATS = 3
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)  # warm-up
            torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            torch.mps.synchronize()
            results[gc] = (time.perf_counter() - t0) / REPEATS * 1000  # ms/iter

        with capsys.disabled():
            print(f"\n[time | MPS | Klein1B]")
            print(f"  plain  (gc=False): {results[False]:.1f} ms/iter")
            print(f"  checkp (gc=True):  {results[True]:.1f} ms/iter")
            overhead = (results[True] - results[False]) / max(results[False], 1e-6) * 100
            print(f"  overhead: {overhead:+.1f}%")


# ---------------------------------------------------------------------------
# Tests — Klein9B model (MPS, local Mac)
# 256x256 image → 16x16 latent grid (L=256), 64 text tokens
# Requires ~36 GB unified memory (float32 weights)
# ---------------------------------------------------------------------------

class TestKlein9BMPSGradientCheckpointing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("mps")
        self.dtype  = torch.float32  # bfloat16 not supported on MPS
        self.params = Klein9BParams()

    def _make_model(self, gradient_checkpointing: bool) -> Flux2:
        model = Flux2(self.params, gradient_checkpointing=gradient_checkpointing)
        return model.to(device=self.device, dtype=self.dtype).train()

    def test_outputs_match(self):
        """GC and non-GC models with identical weights should produce identical outputs."""
        model_plain = self._make_model(gradient_checkpointing=False)
        model_gc    = self._make_model(gradient_checkpointing=True)
        model_gc.load_state_dict(model_plain.state_dict())

        out_plain = run_forward_backward(model_plain, self.device, self.dtype, h=16, w=16, t=64)
        out_gc    = run_forward_backward(model_gc,    self.device, self.dtype, h=16, w=16, t=64)

        assert torch.allclose(out_plain, out_gc, atol=1e-3), (
            f"Max abs diff: {(out_plain - out_gc).abs().max().item():.6f}"
        )

    def test_gradients_exist(self):
        """Both variants should produce non-None gradients after backward."""
        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[gc={gc}] {name} has no gradient"

    def test_memory_profiling(self, capsys):
        """
        Profile and print MPS memory for plain vs gradient-checkpointed forward+backward.
        Uses current_allocated_memory() before/after — torch.mps has no peak memory API,
        so this measures net allocation delta rather than true peak. No assertion.
        """
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            torch.mps.empty_cache()
            before_mb = torch.mps.current_allocated_memory() / 1024 ** 2
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            torch.mps.synchronize()
            after_mb = torch.mps.current_allocated_memory() / 1024 ** 2
            results[gc] = after_mb - before_mb

        with capsys.disabled():
            print(f"\n[memory | MPS (net delta) | Klein9B]")
            print(f"  plain  (gc=False): {results[False]:.1f} MB")
            print(f"  checkp (gc=True):  {results[True]:.1f} MB")
            print(f"  note: torch.mps has no peak-memory API; run CUDA test for true peak")

    def test_time_profiling(self, capsys):
        """
        Profile and print wall-clock time for plain vs gradient-checkpointed forward+backward.
        No strict assertion — GC trades memory for compute time.
        """
        REPEATS = 3
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)  # warm-up
            torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_forward_backward(model, self.device, self.dtype, h=16, w=16, t=64)
            torch.mps.synchronize()
            results[gc] = (time.perf_counter() - t0) / REPEATS * 1000  # ms/iter

        with capsys.disabled():
            print(f"\n[time | MPS | Klein9B]")
            print(f"  plain  (gc=False): {results[False]:.1f} ms/iter")
            print(f"  checkp (gc=True):  {results[True]:.1f} ms/iter")
            overhead = (results[True] - results[False]) / max(results[False], 1e-6) * 100
            print(f"  overhead: {overhead:+.1f}%")


# ---------------------------------------------------------------------------
# Tests — Klein9B model (CUDA)
# 1024x1024 image → 64x64 latent grid (L=4096), 128 text tokens
# ---------------------------------------------------------------------------

class TestKlein9BGradientCheckpointing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        self.dtype  = torch.bfloat16
        self.params = Klein9BParams()

    def _make_model(self, gradient_checkpointing: bool) -> Flux2:
        with torch.device(self.device):
            model = Flux2(self.params, gradient_checkpointing=gradient_checkpointing).to(self.dtype)
        return model.train()

    def test_flag_stored_on_init(self):
        params = Klein9BParams()
        assert Flux2(params, gradient_checkpointing=False).gradient_checkpointing is False
        assert Flux2(params, gradient_checkpointing=True).gradient_checkpointing  is True

    def test_outputs_match(self):
        """GC and non-GC models with identical weights should produce identical outputs."""
        model_plain = self._make_model(gradient_checkpointing=False)
        model_gc    = self._make_model(gradient_checkpointing=True)
        model_gc.load_state_dict(model_plain.state_dict())

        out_plain = run_forward_backward(model_plain, self.device, self.dtype, h=64, w=64)
        out_gc    = run_forward_backward(model_gc,    self.device, self.dtype, h=64, w=64)

        assert torch.allclose(out_plain, out_gc, atol=1e-3), (
            f"Max abs diff: {(out_plain - out_gc).abs().max().item():.6f}"
        )

    def test_gradients_exist(self):
        """Both variants should produce non-None gradients after backward."""
        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=64, w=64)
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[gc={gc}] {name} has no gradient"
            del model
            torch.cuda.empty_cache()

    def test_memory_profiling(self, capsys):
        """
        Profile and print peak CUDA memory for plain vs gradient-checkpointed
        forward+backward. Asserts GC peak memory <= plain peak memory.
        """
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            run_forward_backward(model, self.device, self.dtype, h=64, w=64)
            torch.cuda.synchronize(self.device)
            results[gc] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            del model
            torch.cuda.empty_cache()

        with capsys.disabled():
            print(f"\n[memory | CUDA | Klein9B]")
            print(f"  plain  (gc=False): {results[False]:.1f} MB")
            print(f"  checkp (gc=True):  {results[True]:.1f} MB")
            reduction = (results[False] - results[True]) / max(results[False], 1e-6) * 100
            print(f"  reduction: {reduction:.1f}%")

        assert results[True] <= results[False], (
            f"Expected GC to use <= memory, got {results[True]:.1f} MB vs {results[False]:.1f} MB"
        )

    def test_time_profiling(self, capsys):
        """
        Profile and print wall-clock time for plain vs gradient-checkpointed
        forward+backward. No strict assertion — GC trades memory for compute time.
        """
        REPEATS = 3
        results = {}

        for gc in (False, True):
            model = self._make_model(gradient_checkpointing=gc)
            run_forward_backward(model, self.device, self.dtype, h=64, w=64)  # warm-up
            torch.cuda.synchronize(self.device)

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_forward_backward(model, self.device, self.dtype, h=64, w=64)
            torch.cuda.synchronize(self.device)
            results[gc] = (time.perf_counter() - t0) / REPEATS * 1000  # ms/iter
            del model
            torch.cuda.empty_cache()

        with capsys.disabled():
            print(f"\n[time | CUDA | Klein9B]")
            print(f"  plain  (gc=False): {results[False]:.1f} ms/iter")
            print(f"  checkp (gc=True):  {results[True]:.1f} ms/iter")
            overhead = (results[True] - results[False]) / max(results[False], 1e-6) * 100
            print(f"  overhead: {overhead:+.1f}%")
