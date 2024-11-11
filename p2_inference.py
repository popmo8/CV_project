import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import imageio
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))
    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}
def write_mask(output, path):
    
    # output = output.detach().cpu().numpy()
    mask_img = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_img[np.where(output == 0)] = cls_color[0]
    mask_img[np.where(output == 1)] = cls_color[1]
    mask_img[np.where(output == 2)] = cls_color[2]
    mask_img[np.where(output == 3)] = cls_color[3]
    mask_img[np.where(output == 4)] = cls_color[4]
    mask_img[np.where(output == 5)] = cls_color[5]
    mask_img[np.where(output == 6)] = cls_color[6]
    imageio.imwrite(path, mask_img)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_tfm = transforms.Compose([
    transforms.ToTensor()
])

class P2ADataset(Dataset):
    def __init__(self, path, tfm):
        super(P2ADataset).__init__()
        self.filenames = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith("sat.jpg")])
        self.transform = tfm
        self.len = len(self.filenames)
        self.labels = read_masks(path)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        filename = self.filenames[idx]
        im = Image.open(filename)
        im = self.transform(im)

        return im, self.labels[idx], filename.split('/')[-1]
    
batch_size = 16
valid_set = P2ADataset(input_dir, tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

class Deeplabv3_Mobilenet_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_Mobilenet_Model, self).__init__()
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 7, 1, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output["out"]

num_classes = 7
model = Deeplabv3_Mobilenet_Model().to(device)
checkpoint_path = "./submit_model/p2_b_final.ckpt"
checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

def save_mask():

    for imgs, labels, filenames in valid_loader:

        with torch.no_grad():
            logits = model(imgs.to(device))
        logits = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(len(filenames)):
            predicted_mask = logits[i]
            filename = filenames[i]
            write_mask(predicted_mask, os.path.join(output_dir, filename[:4]+'_mask.png'))

save_mask()


