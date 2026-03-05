import os
from torch import Generator
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from scipy.io import loadmat
from PIL import Image
import torchvision.transforms as transforms

# from torchvision.datasets.utils import download_url, extract_archive

# def download_dataset():
#     image_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
#     labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
    
#     root = "data/flower_data"
#     os.makedirs(root, exist_ok=True)

#     download_url(image_url, root=root)
#     download_url(labels_url, root=root)

#     archive_path = os.path.join(root, "102flowers.tgz")
#     extract_archive(archive_path, root)

class OxfordFLowersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'jpg')
        
        labels_mat = loadmat(os.path.join(root_dir, 'imagelabels.mat'))
        self.labels = labels_mat['labels'][0]-1
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = f'image_{index+1:05d}.jpg'
        image_path = os.path.join(self.img_dir, image_name)
        
        image = Image.open(image_path)
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]
)

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]
)

generator = Generator().manual_seed(42)

dataset = OxfordFLowersDataset('data/flower_data')
# print(f'Total Samples {len(dataset)}')
# img, label = dataset[0]
# print(img, label)

#----------------Transformer Test in Single Batch -----------------------------
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# for images, labels in data_loader:
#    print(f'Succes! Batch Shape : {images.shape}')
#    break 

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_index, val_index, test_index = random_split(dataset, [train_size, val_size, test_size], generator=generator)

dataset_train = Subset(OxfordFLowersDataset('data/flower_data', transform=transform_train), train_index.indices)
dataset_val = Subset(OxfordFLowersDataset('data/flower_data', transform=transform_val), val_index.indices)
dataset_test = Subset(OxfordFLowersDataset('data/flower_data', transform=transform_val), test_index.indices)

# print(f'Training: {len(train_dataset)} images')
# print(f'Validation: {len(val_dataset)} images')
# print(f'Test: {len(test_dataset)} images')

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

#----------One Batch from each
for name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
    images, labels = next(iter(loader))
    print(f'{name}: batch{images.shape}')
    
