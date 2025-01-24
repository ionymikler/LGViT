import torch

# Load the state_dict
state_dict = torch.load('pytorch_model.bin', map_location='cpu')

# Print the keys (layer names and parameter shapes)
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
