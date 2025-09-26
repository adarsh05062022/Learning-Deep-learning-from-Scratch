import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        # Directly collect all image files in the root folder
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        img_path = os.path.join(self.root_dir, img_file)

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]

        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="/data1/adarsh/CELEB_FACE_DATASET/img_align_celeba/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)
