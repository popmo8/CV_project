import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from byol_pytorch import BYOL
import torchvision

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class P1PretrainDataset(Dataset):
    def __init__(self, path, tfm):
        super(P1PretrainDataset).__init__()
        self.filenames = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.transform = tfm
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        im = Image.open(filename)
        im = self.transform(im)
        return im

train_set = P1PretrainDataset("./hw1_data/p1_data/mini/train", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

resnet = torchvision.models.resnet50(weights = None)
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
n_epochs = 300
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=n_epochs)

batch_size = 64
save_every_n_epochs = 50
patience = 15

def train(model, epoch, log_interval=100):
    optimizer = opt
    model.train() 
    
    iteration = 0
    lowest = 100000
    patience_cnt = 0
    for ep in range(epoch):
        ep_loss = 0
        for batch_idx, imgs in enumerate(train_loader):
            loss = model(imgs.to(device))
            ep_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.update_moving_average()
            
            if (iteration+1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(imgs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1
            
        ep_loss /= len(train_loader.dataset)//batch_size
        print(f"Train Epoch {ep} End\tLoss: {ep_loss}")
        if (ep+1)%save_every_n_epochs == 0:
            print(f"Regularly save model to p1_pretrain_{ep+1}.pt")
            torch.save(resnet.state_dict(), f'./hw1_data/model/p1_pretrain_{ep+1}.pt')
        if ep_loss < lowest:
            lowest = ep_loss
            print(f"Find optimal, save model to p1_pretrain_{ep+1}_optimal.pt")
            torch.save(resnet.state_dict(), f'./hw1_data/model/p1_pretrain_{ep+1}_optimal.pt')
            patience_cnt = 0
        else:
            patience_cnt += 1
        if (patience_cnt >= patience):
            print(f"Out of patience, break, save model to p1_pretrain_{ep+1}.pt")
            torch.save(resnet.state_dict(), f'./hw1_data/model/p1_pretrain_{ep+1}.pt')
            break

train(learner, n_epochs)