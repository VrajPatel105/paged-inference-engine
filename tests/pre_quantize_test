import torch

def quantize(x: torch.Tensor):
    """Symmetric INT8 quantization. x: fp16/fp32 tensor. Returns (int8_tensor, scale)."""
    max_abs = x.abs().max()
    if max_abs == 0:
        # all-zero tensor edge case: scale doesn't matter, avoid div-by-zero
        scale = torch.tensor(1.0)
    else:
        scale = max_abs / 127.0
    x_int8 = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return x_int8, scale.item()

def dequantize(x_int8: torch.Tensor, scale: float):
    return x_int8.to(torch.float32) * scale


if __name__ == "__main__":
    torch.manual_seed(0)

    # Test 1: random tensor, typical range
    x = torch.randn(1000) * 2.0
    x_int8, scale = quantize(x)
    x_recon = dequantize(x_int8, scale)
    max_err = (x - x_recon).abs().max().item()
    print(f"Test 1 (random normal): max abs error = {max_err:.6f}, scale = {scale:.6f}")
    assert max_err <= scale + 1e-5, "error exceeds one quantization step"

    # Test 2: all-zero tensor
    x_zero = torch.zeros(100)
    x_int8, scale = quantize(x_zero)
    x_recon = dequantize(x_int8, scale)
    print(f"Test 2 (all zeros): max abs error = {(x_zero - x_recon).abs().max().item():.6f}")
    assert torch.allclose(x_recon, x_zero)

    # Test 3: single large outlier (this is the case B-approximate cares about)
    x_outlier = torch.randn(100) * 0.1
    x_outlier[0] = 50.0  # one big outlier
    x_int8, scale = quantize(x_outlier)
    x_recon = dequantize(x_int8, scale)
    err = (x_outlier - x_recon).abs()
    print(f"Test 3 (outlier present): max abs error = {err.max().item():.6f}, "
          f"error on small values = {err[1:].max().item():.6f}, scale = {scale:.6f}")
    # this test doesn't assert - just observe how much the outlier hurts precision
    # on the small values, since that's the real tradeoff you're accepting

    print("\nAll assertions passed.")


# (mlenv) vraj@Vraj:/mnt/c/dev/projects/paged-inference-engine$ /home/vraj/gpu-work/mlenv/bin/python /mnt/c/dev/projects/paged-inference-engine/tests/pre_quantize_test
# Test 1 (random normal): max abs error = 0.032230, scale = 0.064590
# Test 2 (all zeros): max abs error = 0.000000
# Test 3 (outlier present): max abs error = 0.182031, error on small values = 0.182031, scale = 0.393701

# All assertions passed.