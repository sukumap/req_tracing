import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

try:
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Try a simple CUDA operation
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            y = x + x
            print(f"CUDA tensor test: Success - Device: {x.device}, Value: {y}")
        except Exception as e:
            print(f"CUDA tensor operation failed: {e}")
    else:
        print("CUDA is not available")
except Exception as e:
    print(f"Error accessing CUDA properties: {e}")

# Check GPU memory
try:
    if torch.cuda.is_available():
        print("\nGPU Memory Statistics:")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Reserved memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Free memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB")
except Exception as e:
    print(f"Error checking GPU memory: {e}")

# Try to reset the GPU
try:
    if torch.cuda.is_available():
        print("\nAttempting to reset GPU cache...")
        torch.cuda.empty_cache()
        print("GPU cache cleared")
except Exception as e:
    print(f"Error clearing GPU cache: {e}")
