import torch

loaded_model = torch.load('data/model/mowly/sample_model/final_model.pth')
for key in loaded_model.keys():
    print(key)