import torch

# Load the state_dict
model_path = '/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100/pytorch_model.bin'
state_dict = torch.load(model_path, map_location='cpu')

# Print the keys (layer names and parameter shapes)
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
