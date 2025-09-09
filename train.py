#!/usr/bin/env python3
"""
Clean Kolam Vision Model - Training and Testing
Uses EfficientNet-B4 with all archive folders
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import json
import pandas as pd

# Enhanced configuration with new pattern types
CONFIG = {
    'model': {
        'name': 'efficientnet_b4',
        'num_classes': 5,  # Increased for new dot-grid category
        'input_size': (320, 320),  # Reduced for memory efficiency
        'dropout': 0.3
    },
    'training': {
        'batch_size': 8,  # Reduced for RTX 4050 6GB memory
        'epochs': 25,  # Increased for new data
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'paths': {
        'archive': 'archive',
        'checkpoint': 'kolam_model.pth',
        'results': 'results',
        'csv_data': 'archive/Kolam CSV files/Kolam CSV files'  # CSV coordinate data
    }
}

class KolamDataset(Dataset):
    """Clean dataset implementation for Kolam images"""
    
    def __init__(self, archive_path='archive', transform=None, split='train'):
        self.transform = transform
        self.samples = []
        
        # Enhanced folder to class mapping with new patterns
        self.class_map = {
            'Kolam19 Images': 0,  # Simple (19-dot)
            'Kolam29 Images': 1,  # Intermediate (29-dot) 
            'Kolam109 Images': 2, # Complex (109-dot)
            'scrapped': 3,        # Traditional scraped
            'dot_grid': 4         # Dot-grid patterns (new category)
        }
        
        # Pattern characteristics for each class
        self.pattern_types = {
            0: 'Simple dots with basic lines',
            1: 'Medium complexity with curves',
            2: 'Complex interlocking patterns',
            3: 'Traditional mixed styles',
            4: 'Pure dot-grid arrangements'
        }
        
        # Load all images
        self._load_images(Path(archive_path))
        
        # Load CSV coordinate data if available
        self._load_csv_data()
        
        # Split dataset
        if split != 'all':
            self._split_dataset(split)
        
        print(f"Loaded {len(self.samples)} {split} samples with {len(self.csv_data)} coordinate sets")
    
    def _load_images(self, archive_path):
        """Load all images from archive folders"""
        for folder_name, class_id in self.class_map.items():
            folder_path = archive_path / folder_name
            
            # Handle nested folder structure
            if folder_name != 'scrapped':
                folder_path = folder_path / folder_name
            
            if folder_path.exists():
                for img_file in folder_path.glob('*.jpg'):
                    self.samples.append({
                        'path': str(img_file),
                        'label': class_id,
                        'folder': folder_name
                    })
                for img_file in folder_path.glob('*.png'):
                    self.samples.append({
                        'path': str(img_file),
                        'label': class_id,
                        'folder': folder_name
                    })
    
    def _load_csv_data(self):
        """Load coordinate data from CSV files for enhanced pattern understanding"""
        self.csv_data = {}
        csv_path = Path(CONFIG['paths'].get('csv_data', ''))
        
        if csv_path.exists():
            csv_files = {
                'kolam19.csv': 19,
                'kolam29.csv': 29,
                'kolam109.csv': 109
            }
            
            for csv_file, dot_count in csv_files.items():
                file_path = csv_path / csv_file
                if file_path.exists():
                    try:
                        # Read CSV with coordinate data
                        df = pd.read_csv(file_path)
                        self.csv_data[dot_count] = df
                        print(f"  Loaded {len(df)} coordinate patterns from {csv_file}")
                    except Exception as e:
                        print(f"  Warning: Could not load {csv_file}: {e}")
    
    def _split_dataset(self, split):
        """Split dataset into train/val/test"""
        # Get labels for stratification
        labels = [s['label'] for s in self.samples]
        
        # First split: 80% train+val, 20% test
        train_val_idx, test_idx = train_test_split(
            range(len(self.samples)), 
            test_size=0.2, 
            stratify=labels,
            random_state=42
        )
        
        # Second split: 80% train, 20% val from train_val
        train_val_samples = [self.samples[i] for i in train_val_idx]
        train_val_labels = [s['label'] for s in train_val_samples]
        
        train_idx, val_idx = train_test_split(
            range(len(train_val_samples)),
            test_size=0.2,
            stratify=train_val_labels,
            random_state=42
        )
        
        # Assign samples based on split
        if split == 'train':
            self.samples = [train_val_samples[i] for i in train_idx]
        elif split == 'val':
            self.samples = [train_val_samples[i] for i in val_idx]
        elif split == 'test':
            self.samples = [self.samples[i] for i in test_idx]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

class KolamModel(nn.Module):
    """Clean EfficientNet-based Kolam classifier"""
    
    def __init__(self, config=CONFIG):
        super().__init__()
        
        # Load pre-trained backbone
        self.backbone = timm.create_model(
            config['model']['name'],
            pretrained=True,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, *config['model']['input_size'])
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(256, config['model']['num_classes'])
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_transforms():
    """Get data transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(CONFIG['model']['input_size']),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(CONFIG['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model():
    """Train the enhanced Kolam model with new pattern types"""
    print("\n" + "="*60)
    print(" ENHANCED KOLAM MODEL TRAINING ")
    print("="*60)
    
    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Class names for better logging
    class_names = ['Simple (19-dot)', 'Intermediate (29-dot)', 
                   'Complex (109-dot)', 'Traditional', 'Dot-Grid']
    
    device = torch.device(CONFIG['training']['device'])
    print(f"Using device: {device}")
    print(f"Training for {CONFIG['model']['num_classes']} classes: {', '.join(class_names)}")
    
    # Prepare data
    train_transform, val_transform = get_transforms()
    
    train_dataset = KolamDataset(transform=train_transform, split='train')
    val_dataset = KolamDataset(transform=val_transform, split='val')
    test_dataset = KolamDataset(transform=val_transform, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    model = KolamModel(CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['training']['epochs'])
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['training']['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["training"]["epochs"]}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'config': CONFIG,
                'history': history
            }, CONFIG['paths']['checkpoint'])
            print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
        
        scheduler.step()
        print('-' * 60)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE ")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {CONFIG['paths']['checkpoint']}")

def test_image(image_path):
    """Test a single image"""
    device = torch.device(CONFIG['training']['device'])
    
    # Load model
    model = KolamModel(CONFIG).to(device)
    checkpoint = torch.load(CONFIG['paths']['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare image
    _, transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    # Updated class names with new pattern type
    classes = ['Simple (19-dot)', 'Intermediate (29-dot)', 
              'Complex (109-dot)', 'Traditional', 'Dot-Grid']
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item():.2%}")
    
    # Show top 3 predictions
    top3_prob, top3_classes = torch.topk(probabilities, 3)
    print("\nTop 3 Predictions:")
    for i in range(3):
        print(f"  {i+1}. {classes[top3_classes[0][i]]}: {top3_prob[0][i]:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Clean Kolam Model')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', type=str, help='Test on image')
    parser.add_argument('--batch', type=str, help='Test on folder')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.test:
        if not Path(CONFIG['paths']['checkpoint']).exists():
            print("Model not found. Train first with --train")
        else:
            test_image(args.test)
    elif args.batch:
        if not Path(CONFIG['paths']['checkpoint']).exists():
            print("Model not found. Train first with --train")
        else:
            folder = Path(args.batch)
            for img_file in folder.glob('*.jpg'):
                test_image(str(img_file))
                print()
    else:
        print("Usage:")
        print("  Train model: python kolam_model_clean.py --train")
        print("  Test image:  python kolam_model_clean.py --test image.jpg")
        print("  Test folder: python kolam_model_clean.py --batch folder/")

if __name__ == "__main__":
    main()
