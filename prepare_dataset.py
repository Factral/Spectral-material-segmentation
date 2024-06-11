from PIL import Image, ImageColor
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from spectral import open_image
from constants import BANDS

def process_mask(rgb_mask, colormap):
    """Convert RGB mask to a one-hot encoded mask based on a colormap."""
    output_mask = np.zeros((*rgb_mask.shape[:2], len(colormap)), dtype=bool)

    for i, color in enumerate(colormap):
        output_mask[:, :, i] = np.all(rgb_mask == color, axis=-1)
    
    return output_mask

def inverse_process_mask(mask, colormap):
    """Convert a one-hot encoded mask back to RGB using a colormap."""
    output_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    
    for i, color in enumerate(colormap):
        output_mask[mask[:, :, i]] = color
    
    return output_mask


def main(args):
    """
        We need to convert the RGB masks to one-hot encoded masks and save them back to the disk.
        We also need to downsample the hyperspectral cubes to 31 bands and save them as numpy arrays.
    """
    
    data_dir = Path(args.dir)

    info = json.load(open(data_dir / 'label_info.json'))
    colormap = [ImageColor.getcolor(i['color_hex_code'], 'RGB') for i in info['items']]

    files = [f.stem for f in (data_dir / args.split / "rgb").glob('*.png')]

    for file in tqdm(files):
        label_rgb = np.array(Image.open(data_dir / args.split / "labels" / f"{file}.png"))

        if label_rgb.shape[-1] == 3:
            mask = process_mask(label_rgb, colormap)
            mask = np.argmax(mask, axis=-1)

            Image.fromarray(mask.astype(np.uint8)).save(data_dir / args.split / "labels" / f"{file}.png")

        if not (data_dir / args.split / "reflectance_cubes" / f"{file}.npy").exists():
            cube = open_image(data_dir / args.split / "reflectance_cubes" / f"{file}.hdr").load()
            cube = np.rot90(cube, 3)

            spectral_range = [397.32, 702.58]
            bands = 31

            end = BANDS.index(spectral_range[1])

            bands_in_range = len(BANDS[:end+1])

            resolution = (spectral_range[1] - spectral_range[0]) / bands
            resolution = round(resolution)

            downsampled_cube = np.zeros((cube.shape[0], cube.shape[1], bands))

            visible = [400,700]
            if bands < bands_in_range:
                for band in range(bands):
                    range_spectral = visible[0] + (band * resolution)
                    idx1 = min(BANDS, key=lambda x:abs(x-range_spectral))
                    idx1 = BANDS.index(idx1)
                    if BANDS[idx1] < range_spectral:
                        idx1 = idx1 + 1
                    idx2 = min(BANDS, key=lambda x:abs(x-(range_spectral+resolution)))
                    idx2 = BANDS.index(idx2)
                    if BANDS[idx2] > range_spectral + resolution:
                        idx2 = idx2 - 1

                    downsampled_cube[:,:,band] = np.mean(cube[:,:,idx1:idx2], axis=2)

            np.save(data_dir / args.split / "reflectance_cubes" / f"{file}.npy", downsampled_cube)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare hyprspectral cubes for training.')
    parser.add_argument('--dir', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--split', type=str, help='train, test or val', required=True)

    args = parser.parse_args()

    main(args)




