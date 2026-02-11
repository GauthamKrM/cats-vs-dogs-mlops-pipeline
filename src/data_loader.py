from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32):
    """
    Returns train, val, test dataloaders.
    Applies data augmentation to training set and standard normalization to val/test.
    """
    # Create distinct datasets for training and validation/testing
    # This ensures train_dataset gets augmentation, while val/test do not.
    # We use the same random split indices for consistency.
    
    # 1. Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load datasets with respective transforms
    # ImageFolder loads files sorted by name, so indices will align.
    train_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_test_full_dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transform)
    
    # 3. Calculate split sizes
    total_size = len(train_full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # 4. Generate random indices
    # We use torch.utils.data.random_split behavior manually or use generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    # 5. Create Subsets
    train_set = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_set = torch.utils.data.Subset(val_test_full_dataset, test_indices)
    
    # 6. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test
    # train, val, test = get_dataloaders("data")
    pass
