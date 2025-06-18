import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BTCVDataset(Dataset):
    def __init__(self, root_dir, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        if test:
            pass

        else:
            for organ_dir in os.listdir(self.root_dir):
                self.images = sorted(os.listdir(os.path.join(root_dir, organ_dir, "imagesTr")))
                self.images = sorted(os.listdir(os.path.join(root_dir, organ_dir, "imag")))
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        mask_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask