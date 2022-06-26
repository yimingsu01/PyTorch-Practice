import torch
from PIL import ImageTk, Image
import cv2

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform):
        self.rootDir = rootDir
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['Image_path'][idx]
        image = sk.imread(imagePath)
        label = self.data['Condition'][idx]
        image = Image.fromarray(image)

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label