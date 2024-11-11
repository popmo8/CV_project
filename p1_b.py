import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.nn as nn
import wandb

wandb.init(project="pytorch-example")

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class P1BDataset(Dataset):
    def __init__(self, path, tfm):
        super(P1BDataset).__init__()
        self.filenames = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.transform = tfm
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        filename = self.filenames[idx]
        im = Image.open(filename)
        im = self.transform(im)
        try:
            label = int(filename.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im, label

resnet = torchvision.models.resnet50(weights = None)
resnet.load_state_dict(torch.load('./hw1_data/p1_data/pretrain_model_SL.pt'))
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 65)
model = resnet.to(device)

batch_size = 64
n_epochs = 200

train_set = P1BDataset("./hw1_data/p1_data/office/train", tfm=TRANSFORM_IMG)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_set = P1BDataset("./hw1_data/p1_data/office/val", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0003
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, steps_per_epoch=len(train_loader), epochs=n_epochs)

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

def train(model, epoch, log_interval=20):
    iteration = 0
    patience_cnt = 0
    for ep in range(epoch):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            wandb.log({"train acc": acc, "train loss": loss})
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                    ep, batch_idx * len(imgs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc))
            iteration += 1
        
        print(f'Train Epoch: {ep} End')
        model.eval()  # Important: set evaluation mode
        test_loss = 0
        correct = 0
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(valid_loader.dataset)
        test_acc = 100. * correct / len(valid_loader.dataset)
        wandb.log({"test loss": test_loss, "test acc": test_acc})
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(valid_loader.dataset),
            test_acc))
        

train(model, n_epochs)

test(model)

wandb.finish()

torch.save(model.state_dict(), './hw1_data/model/p1_b_acc55.pt')

