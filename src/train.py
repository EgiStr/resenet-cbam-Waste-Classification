import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import os
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from models.cbam_resnet import ResNet34

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 2  # Binary: organic vs recyclable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming data is organized as data/train/organic, data/train/recyclable, etc.
# For Mendeley Waste Dataset, download and organize accordingly
# For demo, using placeholder paths with error handling
try:
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='data/val', transform=val_transform)
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Classes: {train_dataset.classes}")
except FileNotFoundError as e:
    logger.error(f"Data directory not found: {e}")
    logger.error("Please ensure data is organized as:")
    logger.error("data/train/organic/ and data/train/recyclable/")
    logger.error("data/val/organic/ and data/val/recyclable/")
    logger.error("Or run src/download_data.py and src/prepare_data.py first")
    raise e

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model
model = ResNet34(num_classes=NUM_CLASSES)
model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Training function
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1} Training')
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100 * correct / total
    epoch_time = time.time() - start_time
    
    logger.info(f'Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, Time={epoch_time:.2f}s')
    return epoch_loss

# Validation function
def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1} Validation')
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_time = time.time() - start_time
    
    logger.info(f'Val Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}, Time={epoch_time:.2f}s')
    return epoch_loss, accuracy, f1, all_preds, all_labels

# Main training loop
best_f1 = 0.0
train_losses = []
val_losses = []
val_accs = []
val_f1s = []

logger.info("Starting training...")
logger.info(f"Model: CBAM-ResNet34, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
logger.info(f"Device: {DEVICE}, Optimizer: Adam(lr={LEARNING_RATE}), Scheduler: StepLR")

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, DEVICE, epoch)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'models/best_model.pth')
        logger.info(f'New best model saved with F1: {best_f1:.4f}')

logger.info('Training completed!')
logger.info(f'Best F1-Score: {best_f1:.4f}')

# Final evaluation with detailed metrics
logger.info("Performing final evaluation...")
_, _, _, all_preds, all_labels = validate_epoch(model, val_loader, criterion, DEVICE, -1)

# Classification report
report = classification_report(all_labels, all_preds, target_names=['Organic', 'Recyclable'], output_dict=True)
logger.info("Classification Report:")
logger.info(f"Accuracy: {report['accuracy']:.4f}")
logger.info(f"Precision (Organic): {report['Organic']['precision']:.4f}")
logger.info(f"Recall (Organic): {report['Organic']['recall']:.4f}")
logger.info(f"F1 (Organic): {report['Organic']['f1-score']:.4f}")
logger.info(f"Precision (Recyclable): {report['Recyclable']['precision']:.4f}")
logger.info(f"Recall (Recyclable): {report['Recyclable']['recall']:.4f}")
logger.info(f"F1 (Recyclable): {report['Recyclable']['f1-score']:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
logger.info(f"Confusion Matrix:\n{cm}")

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.subplot(1, 3, 3)
plt.plot(val_f1s, label='Val F1-Score')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()
plt.title('Validation F1-Score')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
logger.info("Training curves saved to training_curves.png")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Organic', 'Recyclable'], 
            yticklabels=['Organic', 'Recyclable'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
logger.info("Confusion matrix saved to confusion_matrix.png")

plt.show()