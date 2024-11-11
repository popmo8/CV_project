import sys
import pandas as pd
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os

img_csv_path = sys.argv[1]
img_folder_path = sys.argv[2]
output_path_file = sys.argv[3]

df = pd.read_csv(img_csv_path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

resnet = torchvision.models.resnet50(weights = None)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 65)
model = resnet.to(device)
model.load_state_dict(torch.load('./submit_model/p1_c.pt', weights_only=True))
model.eval()

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for index, filename in df['filename'].items():
    im = Image.open(os.path.join(img_folder_path, filename))
    im = test_tfm(im).to(device).unsqueeze(0)
    output = model(im)
    pred = output.max(1, keepdim=True)[1].item()
    df.at[index, 'label'] = pred

df.to_csv(output_path_file, index=False)
