from model import SimpleVGG16
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

TRAIN_DATA_PATH = r"D:\PythonProjects\emotion_recognition\dataset\train"
TEST_DATA_PATH = r"D:\PythonProjects\emotion_recognition\dataset\test"
EPOCHS = 30
INIT_LR = 1e-3
BS = 32

train_dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

train_images, train_labels = next(iter(train_dataloader))

test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

vgg = SimpleVGG16()

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(vgg.parameters(), lr=0.0001)

classes = train_dataset.classes
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(train_dataloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BS)))

for epoch in range(2):  # loop over the dataset multiple times
    print(f"epoch {epoch}")

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(f"length: {len(inputs)}")

        # zero the parameter gradients
        opt.zero_grad()
        print("grad zeroed")

        # forward + backward + optimize
        outputs = vgg(inputs)
        print("forward backward opt")
        loss = criterion(outputs, labels)
        print("forward backward opt")
        loss.backward()
        print("forward backward opt")
        opt.step()
        print("forward backward opt")

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')



