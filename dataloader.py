
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from pathlib import Path

class FacadeDataset(Dataset):
    """Dataset for loading facade images and associated data for segmentation.

    Attributes:
        dir (Path): Directory containing dataset files.
        files (list): List of file names (without extension) in the RGB directory.
        validation (list): List of file names for validation, if applicable.
        transform (callable, optional): Albumentations transform for data augmentation.
    """


    def __init__(self, dir, aug_transform=None):
        """Initialize the dataset with the directory path and optional transform.

        Args:
            dir_path (str or Path): Path to the dataset directory.
            aug_transform (callable, optional): An albumentations transform or composite.
        """
        self.dir = dir # dir must be a Path object
        self.files = [f.stem for f in (dir / "rgb").glob('*.png')]
        self.transform = aug_transform

        if 'train' in str(self.dir):
            dir_temp = str(self.dir).replace('train', '')
            self.validation = [f.stem for f in (Path(dir_temp) / "validation" / "rgb").glob('*.png')]
            self.files += self.validation

        print(f"len of files: {len(self.files)}")


    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.files)


    def __getitem__(self, idx):
        """Get dataset item by index. (structure for pytorch)

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the cube, image, and label tensors.
        """

        # Determine the paths based on the training/validation split.
        if 'train' in str(self.dir):
            if self.files[idx] in self.validation:
                label = self.dir.parent / "validation" / "labels" / (self.files[idx] + ".png")
                hscube = self.dir.parent / "validation" / "reflectance_cubes" / (self.files[idx] + ".npy")
                rgb = self.dir.parent / "validation" / "rgb" / (self.files[idx] + ".png")
                depth = self.dir.parent / "validation" / "depth" / (self.files[idx] + ".png")
            else:
                depth = self.dir / "depth" / (self.files[idx] + ".png")
                label = self.dir / "labels" / (self.files[idx] + ".png")
                hscube = self.dir / "reflectance_cubes" / (self.files[idx] + ".npy")
                rgb = self.dir / "rgb" / (self.files[idx] + ".png")
        else:
            depth = self.dir / "depth" / (self.files[idx] + ".png")
            label = self.dir / "labels" / (self.files[idx] + ".png")
            hscube = self.dir / "reflectance_cubes" / (self.files[idx] + ".npy")
            rgb = self.dir / "rgb" / (self.files[idx] + ".png")


        # Load data from files.
        label = np.array(Image.open(label))
        rgb = np.array(Image.open(rgb))
        cube = np.load(hscube)
        depth = np.array(Image.open(depth))


        if self.transform:
            transformed = self.transform(image=rgb, image0=cube, mask=label, depth=depth)
            rgb = transformed['image']
            cube = transformed['image0']
            label = transformed['mask']
            depth = transformed['depth']


        rgb = torch.from_numpy(np.array(rgb)).float() / 255.0
        rgb = rgb.permute(2, 0, 1)
        
        depth = torch.from_numpy(depth).float() / 255.0
        
        rgbd = torch.cat((rgb, depth.unsqueeze(0)), dim=0)
        
        cube = torch.from_numpy(cube).float()
        cube.clamp_(0 + 1e-6, 1 - 1e-6)
        cube = cube.permute(2, 0, 1)

        label = torch.from_numpy(label).long()

        return cube, rgbd, label.squeeze()
