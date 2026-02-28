import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Optional: Run a test tensor on the GPU
    x = torch.randn(3, 3)
    x = x.to('cuda')
    print("Test Tensor on GPU:")
    print(x)
else:
    print("PyTorch is running on CPU.")
