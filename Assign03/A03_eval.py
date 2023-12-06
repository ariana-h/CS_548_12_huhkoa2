import torch_fidelity as torf
from torch.utils.data import Dataset
import os
from torchvision import transforms
import torch
import cv2

def main():
    metrics = torf.calculate_metrics(
        input1="Assign03/gen_images",
        input2="Assign03/train_images",
        fid=True,
        kid=True,
        kid_subset_size=200,
        cuda=True)


if __name__ == "__main__":
    main()