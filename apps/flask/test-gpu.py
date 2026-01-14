import torch

print("=== GPU Detection ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        print(f"\nGPU {i}: {name}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Cek apakah RTX 3050
        if "RTX 3050" in name or "NVIDIA" in name:
            print(f"  ✅ This is your RTX 3050!")