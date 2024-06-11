import torch
import torch.nn as nn
from PIL import Image, ImageColor
from torchvision import transforms
from architecture.FPN import hrnet_fpn
from pathlib import Path
import argparse 
from utils import process_statedict_dataparallel
import numpy as np
import json

parser = argparse.ArgumentParser(description="Inference over the trained model")
parser.add_argument("--weights", type=str, help='Path to the model weights')
parser.add_argument("--folder_path", type=str, help='Path to folder with images to predict')
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

info = json.load(open(Path('LIB-HSI') / 'label_info.json'))
colormap = [ImageColor.getcolor(i['color_hex_code'], 'RGB') for i in info['items']]


def inverse_process_mask(mask, colormap):
    """Convert categorical mask to RGB color mask."""
    output_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        output_mask[mask == i] = color
    return output_mask


files = [f for f in (Path(args.file)).glob('*.png')]

for file in files:

    rgb = np.array(Image.open(file))
    rgb = torch.from_numpy(np.array(rgb)).float() / 255.0
    image = rgb.permute(2, 0, 1)

    depth = file.parent / "depth" / file.name
    depth = torch.from_numpy(depth).float() / 255.0
    
    image = torch.cat((rgb, depth.unsqueeze(0)), dim=0)


    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = nn.functional.softmax( output , dim=1 )
        prediction = output.argmax(1)


    prediction = prediction.squeeze().cpu().numpy()
    prediction = inverse_process_mask(prediction, colormap)

    im = Image.fromarray(prediction.astype(np.uint8))
    im.save(f"experiments/preds/{file.stem}_pred.png")
    print(f"Saved {file.stem}_pred.png")
