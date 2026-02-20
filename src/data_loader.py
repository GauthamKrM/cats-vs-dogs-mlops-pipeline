from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates and returns train, validation, and test DataLoaders.
    - Training set uses data augmentation.
    - Validation and test sets use only resizing + normalization.
    """

    # -------------------------
    # 1. Define image transforms
    # -------------------------

    # Training transform:
    # Includes random horizontal flip for augmentation
    # Helps model generalize better
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to fixed size (needed for most pretrained models)
        transforms.RandomHorizontalFlip(),  # Random flip for augmentation
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(   # Normalize using ImageNet mean/std (common for pretrained models)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation/Test transform:
    # No augmentation here — we want consistent evaluation
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------
    # 2. Load full datasets
    # -------------------------

    # We load the dataset twice:
    # - One with augmentation (for training subset)
    # - One without augmentation (for val/test subsets)
    # This ensures validation/test data is not randomly altered.
    train_full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    val_test_full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=val_test_transform
    )

    # -------------------------
    # 3. Compute split sizes
    # -------------------------

    total_size = len(train_full_dataset)

    # 80% train, 10% val, 10% test
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # Remaining samples

    # -------------------------
    # 4. Generate shuffled indices
    # -------------------------

    # We manually shuffle indices for reproducibility
    # Using fixed seed ensures same split every time
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()

    # Split indices into train/val/test
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # -------------------------
    # 5. Create dataset subsets
    # -------------------------

    # Subset allows us to use only selected indices
    train_set = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_set = torch.utils.data.Subset(val_test_full_dataset, test_indices)

    # -------------------------
    # 6. Create DataLoaders
    # -------------------------

    # Training loader shuffles batches every epoch
    # Validation/Test loaders should not shuffle
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage:
    # train_loader, val_loader, test_loader = get_dataloaders("data")
    pass