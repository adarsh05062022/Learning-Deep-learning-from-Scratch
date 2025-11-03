import os
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
import torch


# -----------------------
# Helper: show grid of images
# -----------------------
from torchvision.utils import save_image

class ImageFolderFlat(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, extensions=(".jpg", ".jpeg", ".png", ".webp")):
        self.paths = []
        for ext in extensions:
            self.paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img

def save_images(x, nrow=8, title=None, save_dir="results", filename="sample.png"):
    """
    x: BCHW in [-1,1]. Uses torchvision's save_image with normalize+value_range.
    """
    os.makedirs(save_dir, exist_ok=True)
    if title:
        filename = title.replace(" ", "_") + ".png"
    save_path = os.path.join(save_dir, filename)

    # save_image will scale from [-1,1] to [0,1] when normalize=True & value_range specified
    save_image(x, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"✅ Image saved at: {save_path}")

