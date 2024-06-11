import torch
import torch.nn as nn
from PIL import Image, ImageColor
from torchvision import transforms
from architecture.FPN import hrnet_fpn
from pathlib import Path
import argparse 
import matplotlib.pyplot as plt
from utils import process_statedict_dataparallel
import numpy as np
import json
from metrics import Metrics

parser = argparse.ArgumentParser(description="Evaluate the model on the LIB-HSI dataset")
parser.add_argument("--weights", type=str, help='Path to the model weightse')
parser.add_argument("--path_dataset", type=str, help='Path to the dataset')
parser.add_argument("--save_predictions", type=bool, default=False, help='Save the predictions')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict = torch.load(args.weights, map_location=device)

# this is neccesary if the model was trained with DataParallel 
# if not, comment the next line
state_dict = process_statedict_dataparallel(state_dict)

model = hrnet_fpn().to(device)
model.load_state_dict(state_dict)
model.eval()

preprocess = transforms.Compose([
   transforms.ToTensor(),
])


files = [f.stem for f in (Path(args.path_dataset) / "rgb").glob('*.png')]
info = json.load(open(Path('LIB-HSI') / 'label_info.json'))
colormap = [ImageColor.getcolor(i['color_hex_code'], 'RGB') for i in info['items']]


def inverse_process_mask(mask, colormap):
    """Convert categorical mask to RGB color mask."""
    output_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        output_mask[mask == i] = color
    return output_mask

metric_test = Metrics('test')

for file in files:
    rgb_path = Path(args.path_dataset) / "rgb" / (file + ".png")
    depth_path = Path(args.path_dataset) / "depth" / (file + ".png")
    label = Path(args.path_dataset) / "labels" / (file + ".png")

    label = np.array(Image.open(label))
    rgb = np.array(Image.open(rgb_path))
    depth = np.array(Image.open(depth_path))

    # preprocess
    rgb = torch.from_numpy(np.array(rgb)).float() / 255.0
    image = rgb.permute(2, 0, 1)
    depth = torch.from_numpy(depth).float() / 255.0
    
    image = torch.cat((rgb, depth.unsqueeze(0)), dim=0)
    
    label = torch.from_numpy(label).long().unsqueeze(0)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = nn.functional.softmax( output , dim=1 )
        prediction = output.argmax(1)

    metric_test.update(prediction.long() , label.long())

    if args.save_predictions:
        prediction = prediction.squeeze().cpu().numpy()
        prediction = inverse_process_mask(prediction, colormap)
        im = Image.fromarray(prediction.astype(np.uint8))
        im.save(f"results/{file}.png")

pixel_acc, macc, miou = metric_test.compute()

print(f"Pixel accuracy: {pixel_acc}")
print(f"Mean accuracy: {macc}")
print(f"Mean IoU: {miou}")



"""
    useful code if you want to visualize the results of the last prediction

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0), vmin=0, vmax=46)
    plt.title("RGB Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(label.squeeze(), vmin=0, vmax=46)
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze().cpu(), vmin=0, vmax=46)
    plt.title("Prediction")
    plt.axis('off')

    # save plot
    plt.savefig(f"results/{f}.png")
"""








def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(image)
        prediction = output.squeeze().argmax(0)

    return prediction

