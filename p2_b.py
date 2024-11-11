import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import wandb
from mean_iou_evaluate import *
import torchvision.models as models
from tqdm.auto import tqdm


wandb.init(project="dlcv")

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_tfm = transforms.Compose([
    transforms.ToTensor()
])
train_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class P2ADataset(Dataset):
    def __init__(self, path, tfm):
        super(P2ADataset).__init__()
        self.filenames = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.transform = tfm
        self.len = len(self.filenames)
        self.labels = read_masks(path)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        filename = self.filenames[idx]
        im = Image.open(filename)
        im = self.transform(im)

        return im, self.labels[idx]
    
batch_size = 16
train_set = P2ADataset("./hw1_data/p2_data/train", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_set = P2ADataset("./hw1_data/p2_data/validation", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

class Deeplabv3_Mobilenet_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_Mobilenet_Model, self).__init__()
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

        self.model.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 7, 1, 1),
        )
        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output["out"]

num_classes = 7
model = Deeplabv3_Mobilenet_Model().to(device)
print(model)


criterion = nn.CrossEntropyLoss()
learning_rate = 0.0003
weight_decay = 1e-5
n_epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

patience = 200

def train(model, dataloader, optimizer, criterion, patience, num_epochs):
    opt_iou = 0
    patience_cnt = 0
    for epoch in range(num_epochs):
        # Train
        model.train()
        loss_sum = 0
        outputs = np.zeros((len(train_loader), 512, 512))
        masks = np.zeros((len(train_loader), 512, 512))
        cnt = 0
        for batch in tqdm(dataloader):
            images, labels = batch
            optimizer.zero_grad()

            logits = model(images.to(device))
            labels = labels.long().to(device)
            
            loss = criterion(logits, labels)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()

            logits = torch.argmax(logits, dim=1).cpu()
            labels = labels.squeeze(1).cpu()
            outputs[cnt, :, :] = logits[0]
            masks[cnt, :, :] = labels[0]
            cnt += 1

        iou_score = mean_iou_score(outputs, masks)
        wandb.log({"train iou": iou_score, "train loss": loss_sum/len(dataloader)})
        print(f"Epoch [{epoch+1}/{num_epochs}] Train, Loss: {loss_sum/len(dataloader):.4f}, IOU: {iou_score:.4f}")

        # Test
        model.eval()

        valid_loss = 0

        outputs = np.zeros((len(valid_loader), 512, 512))
        masks = np.zeros((len(valid_loader), 512, 512))
        idx = 0

        for batch in tqdm(valid_loader):

            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            labels = labels.long().to(device)
            loss = criterion(logits, labels)
            valid_loss += loss.item()
            

            logits = torch.argmax(logits, dim=1).cpu()
            labels = labels.squeeze(1).cpu()
            outputs[idx, :, :] = logits[0]
            masks[idx, :, :] = labels[0]
            idx += 1
            

        val_iou_score = mean_iou_score(outputs, masks)
        wandb.log({"test iou": val_iou_score, "epoch": epoch, "test loss": valid_loss/len(valid_loader)})
        print(f"Epoch [{epoch+1}/{num_epochs}] Valid, Loss: {valid_loss/len(valid_loader):.4f}, IOU: {val_iou_score:.4f}")

        if val_iou_score > opt_iou:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"./hw1_data/model/p2_b_batch4_{epoch}_optimal.ckpt") # only save best to prevent output memory exceed error
            opt_iou = val_iou_score
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt > patience:
                print("Out of patience, Exit.")
                break

train(model, train_loader, optimizer, criterion, patience, n_epochs)


