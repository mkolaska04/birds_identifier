import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONFIGURATION ---
CUB_DATASET_PATH = "./CUB_200_2011"
IMAGE_SIZE = 224
BATCH_SIZE = 16

class CUBDataset(Dataset):
    """Custom PyTorch Dataset for CUB-200-2011."""
    def __init__(self, dataset_path, is_train=True, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        
        images_txt_path = os.path.join(dataset_path, 'images.txt')
        labels_txt_path = os.path.join(dataset_path, 'image_class_labels.txt')
        split_txt_path = os.path.join(dataset_path, 'train_test_split.txt')

        df_images = pd.read_csv(images_txt_path, sep=' ', names=['img_id', 'filepath'])
        df_labels = pd.read_csv(labels_txt_path, sep=' ', names=['img_id', 'class_id'])
        df_split = pd.read_csv(split_txt_path, sep=' ', names=['img_id', 'is_training_img'])

        df = df_images.merge(df_labels, on='img_id').merge(df_split, on='img_id')
        
        self.df = df[df['is_training_img'] == (1 if is_train else 0)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_path, 'images', row['filepath'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['class_id'] - 1, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """Creates and returns DataLoaders for train, validation, and test sets."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_train_dataset = CUBDataset(CUB_DATASET_PATH, is_train=True, transform=data_transforms['train'])
    test_dataset = CUBDataset(CUB_DATASET_PATH, is_train=False, transform=data_transforms['test'])

    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    classes_txt_path = os.path.join(CUB_DATASET_PATH, 'classes.txt')
    df_classes = pd.read_csv(classes_txt_path, sep=' ', names=['class_id', 'class_name'])
    class_names = [name.split('.')[1].replace('_', ' ') for name in df_classes['class_name']]

    print(f"Data loaded: {len(train_subset)} train, {len(val_subset)} validation, {len(test_dataset)} test images.")
    return train_loader, val_loader, test_loader, class_names