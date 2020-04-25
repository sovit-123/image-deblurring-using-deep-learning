import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import albumentations

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# helper functions
image_dir = '../outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)
    
def save_decoded_image(img, name):
    # img = np.clip(img, 0., 1.)
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

batch_size = 2

gauss_blur = os.listdir('../input/gaussian_blurred/')
gauss_blur.sort()
sharp = os.listdir('../input/sharp')
sharp.sort()

# plt.figure(figsize=(15, 12))
# for i in range(3):
#     blur_image = plt.imread(f"../input/motion_blurred/{mot_blur[i]}")
#     blur_image = cv2.resize(blur_image, (224, 224))
#     plt.subplot(1, 3, i+1)
#     plt.imshow(blur_image)
# plt.show()

# plt.figure(figsize=(15, 12))
# for i in range(3):
#     sharp_image = plt.imread(f"../input/sharp/{sharp[i]}")
#     sharp_image = cv2.resize(sharp_image, (224, 224))
#     plt.subplot(1, 3, i+1)
#     plt.imshow(sharp_image)
# plt.show()

x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])
    
print(x_blur[10])
print(y_sharp[10])

(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

print(len(x_train))
print(len(x_val))

# blur = plt.imread(f"../input/motion_blurred/{x_train[0]}")
# plt.imshow(blur)
# plt.show()

# sharp = plt.imread(f"../input/sharp/{y_train[0]}")
# plt.imshow(sharp)
# plt.show()

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
])

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_image = cv2.imread(f"../input/gaussian_blurred/{self.X[i]}")
        
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(f"../input/sharp/{self.y[i]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)
 
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# the autoencoder network
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder layers
        self.enc1 = nn.Conv2d(3, 128, kernel_size=5, padding=1)
        self.enc2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.enc3 = nn.Conv2d(32, 1, kernel_size=1, padding=1)
        # self.enc3 = nn.Conv2d(256, 128, kernel_size=1)
        # self.enc4 = nn.Conv2d(128, 64, kernel_size=1)
        self.enc5 = nn.Conv2d(64, 3, kernel_size=1, padding=1)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=5, padding=1)
        # self.dec3 = nn.ConvTranspose2d(64, 128, kernel_size=1)
        # self.dec4 = nn.ConvTranspose2d(128, 256, kernel_size=1)
        # self.dec5 = nn.ConvTranspose2d(256, 512, kernel_size=1)
        self.out = nn.ConvTranspose2d(128, 3, kernel_size=1)

    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        # x = F.relu(self.enc3(x))
        # x = F.relu(self.enc3(x))
        # x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        
        # decode
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = (self.out(x))

        return x

model = Autoencoder().to(device)
print(model)

# the loss function
criterion = nn.MSELoss()
# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
        verbose=True
    )

def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")
    
#     if epoch % 10 == 0:
    # save_decoded_image(outputs.cpu().data, name=f"../outputs/saved_images/train_deblurred{epoch}.jpg")
    return train_loss

# the training function
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            if epoch == 0 and i == (len(val_data)/dataloader.batch_size)-1:
                save_decoded_image(sharp_image.cpu().data, name=f"../outputs/saved_images/sharp{epoch}.jpg")
                save_decoded_image(blur_image.cpu().data, name=f"../outputs/saved_images/blur{epoch}.jpg")

        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        # if i == 1:
        #     save_decoded_image(sharp_image.cpu().data, name=f"../outputs/saved_images/sharp{epoch}.jpg")
        save_decoded_image(outputs.cpu().data, name=f"../outputs/saved_images/val_deblurred{epoch}.jpg")
        
        return val_loss

train_loss  = []
val_loss = []
start = time.time()
for epoch in range(50):
    print(f"Epoch {epoch+1} of {50}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()

# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), '../outputs/model.pth')