# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/09-normalizing-flows.html

import lightning
import os
import torch
import torchvision
import urllib.request

import filesystem as fs
import image

# Get git root directory
git_root = fs.git_root()

# Setting the seed
lightning.seed_everything(seed=42)

# Fetching the device that will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST). Create
# the directory if it does not exist yet.
DATA_PATH = os.path.join(git_root, "data")
os.makedirs(name=DATA_PATH, exist_ok=True)

# Path to the folder where the pretrained models are saved. Create the directory if it
# does not exist yet.
CHECKPOINT_PATH = os.path.join(git_root, "saved_models")
os.makedirs(name=CHECKPOINT_PATH, exist_ok=True)

# GitHub URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial11/"

# Files to download
pretrained_files = [
    "MNISTFlow_simple.ckpt",
    "MNISTFlow_vardeq.ckpt",
    "MNISTFlow_multiscale.ckpt",
]

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(path=file_path):
        file_url = os.path.join(base_url, file_name)
        print(f"Downloading {file_url} ...")
        try:
            urllib.request.urlretrieve(url=file_url, filename=file_path)
        except urllib.error.HTTPError as e:
            print(f"Error: Failed to download {file_url}: {e}")
    else:
        print(f"File {file_path} already exists.")

# Transformations applied on each image => make them a tensor and discretize
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), image.discretize]
)

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, transform=transform, download=True
)
