import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.current_device())  # Prints the current CUDA device (if available)
print(torch.cuda.get_device_name(0))  # Prints the name of your CUDA-enabled GPU
