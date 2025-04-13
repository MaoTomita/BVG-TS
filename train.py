import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import wandb
import timm
from BVG_LS import BVG_LS 

# Prepare dataset and DataLoader
# This function creates training and test loaders for the specified dataset.
def prepare_data(dataset_name, batch_size):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == "pets":
        train_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=train_transform)
        test_dataset = datasets.OxfordIIITPet(root="./data", split="test", download=True, transform=test_transform)
    elif dataset_name == "dtd":
        train_dataset = datasets.DTD(root="./data", split="train", download=True, transform=train_transform)
        test_dataset = datasets.DTD(root="./data", split="test", download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Prepare the model and parameter groups for optimization
# This function initializes the model and organizes its parameters into groups for fine-tuning.
def prepare_model(network, num_classes):

    if network == "wrn_50_2":
        model = models.wide_resnet50_2(weights='DEFAULT')
        # Load pretrained weights
        # model.load_state_dict(torch.load("wrn_50_2_pretrained.pth"))
        # Modify the fully connected layer for the specific number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Group layers for parameter optimization
        layers = [
            nn.Sequential(model.conv1, model.bn1),
            model.layer1, model.layer2, model.layer3, model.layer4, model.fc
        ]
        param_groups = [{'params': layer.parameters()} for layer in layers]
        
    elif network == "vit_small":
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True, num_classes=num_classes)
        param_groups = [
            {'params': [model.cls_token, model.pos_embed] + list(model.patch_embed.parameters())},
            *[{'params': block.parameters()} for block in model.blocks],
            {'params': list(model.norm.parameters()) + list(model.head.parameters())}
        ]

    else:
        raise ValueError(f"Unsupported network: {network}")

    return model, param_groups

# Training loop function
# Handles training the model using the BVG_LS fine-tuning strategy.
def train(model, train_loader, test_loader, param_groups, device, epochs=50, batch_size=128, lr=0.01, use_wandb=False):

    model = model.to(device)
    
    # Initialize optimizer and BVG_LS
    optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, dampening=0, weight_decay=5e-4, nesterov=True)
    bvg_ls = BVG_LS(optimizer)

    criterion = nn.CrossEntropyLoss()

    if use_wandb:
        wandb.init(project="github_test", config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        })

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        layer_update_counts = [0] * len(param_groups)  # Track update counts for each layer

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update learning rates using BVG_LS
            bvr, update_layer_idx = bvg_ls.set_lr(global_lr=lr, n_update=1)
            
            # Update parameters
            optimizer.step()

            # Track the update counts for each layer
            for idx in update_layer_idx:
                layer_update_counts[idx] += 1

            running_loss += loss.item()

        # Log training metrics
        if use_wandb:
            wandb.log(data={
                "loss": running_loss / len(train_loader),
                "bvr": {f"layer_{idx}": bvr[idx].item() for idx in range(len(bvr))},
                "update_freq": {f"layer_{idx}": layer_update_counts[idx] / len(train_loader) for idx in range(len(layer_update_counts))}
            }, step=epoch+1)

        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {running_loss / len(train_loader):.4f}")
        validate(model, test_loader, criterion, device, use_wandb, epoch)

# Validation loop function
# Evaluates the model on the test dataset.
def validate(model, test_loader, criterion, device, use_wandb, epoch):

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")

    if use_wandb:
        wandb.log(data={"test_loss": test_loss / len(test_loader), "test_accuracy": accuracy}, step=epoch+1)

# Main function
def main():

    parser = argparse.ArgumentParser(description="Finetuning a model with BVG_LS")
    parser.add_argument("--dataset", type=str, choices=["pets", "dtd", "cifar100"], required=True, help="Target dataset")
    parser.add_argument("--network", type=str, choices=["wrn_50_2", "vit_small"], required=True, help="network")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "pets":
        num_classes = 37
    elif args.dataset == "dtd":
        num_classes = 47
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Invalid dataset")

    model, param_groups = prepare_model(args.network, num_classes)
    train_loader, test_loader = prepare_data(args.dataset, args.batch_size)
    
    train(model, train_loader, test_loader, param_groups, device, args.epochs, args.batch_size, args.lr, args.use_wandb)
    
    print("Finished!")

if __name__ == "__main__":
    main()
