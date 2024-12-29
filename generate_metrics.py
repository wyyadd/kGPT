import os
import pickle

import torch
from tqdm import tqdm

from utils import compute_ade

if __name__ == "__main__":
    predicted_folder = "./data//scenarios"
    target_folder = "./data/validation/processed"

    total_ade = torch.tensor([])
    total_min_ade = torch.tensor([])

    predicted_files = [name for name in os.listdir(predicted_folder)]
    for predicted_file in tqdm(predicted_files):
        with open(os.path.join(predicted_folder, predicted_file), "rb") as f:
            predicted = pickle.load(f)
        with open(os.path.join(target_folder, predicted_file), "rb") as f:
            target = pickle.load(f)
        ade, min_ade = compute_ade(next(iter(predicted.values())), target["agent"])
        total_ade = torch.cat((total_ade, ade.unsqueeze(-1)), dim=0)
        total_min_ade = torch.cat((total_min_ade, min_ade.unsqueeze(-1)), dim=0)
        # print(f"{predicted_file}: ADE: {ade:.4f}, Min-ADE: {min_ade:.4f}")
    print(total_ade.shape, total_min_ade.shape, total_ade.mean(), total_min_ade.mean())
