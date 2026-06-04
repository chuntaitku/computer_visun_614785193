"""
Course: Introduction to Deep Learning for Computer Vision / Computer Vision
Project: Fine-Grained Dog Breed Classification (Stanford Dogs Dataset)
Tech Stack: PyTorch Conversion (Optimized for Ubuntu Linux)
Author: 盧春泰
"""

import os
import cv2
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg') # Saves plots to disk safely over SSH/Headless sessions
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU Active: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# --- HYPERPARAMETERS ---
BASE_DIR = "."
IMAGES_DIR = os.path.join(BASE_DIR, "Images") 
TRAIN_MAT = os.path.join(BASE_DIR, "train_list.mat")
TEST_MAT = os.path.join(BASE_DIR, "test_list.mat")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5  

# =====================================================================
# 1. DATASET DEFINITION & PIPELINES
# =====================================================================
class StanfordDogsDataset(Dataset):
    def __init__(self, mat_path, img_dir, transform=None):
        mat_data = scipy.io.loadmat(mat_path)
        self.img_dir = img_dir
        self.transform = transform
        self.file_list = [str(f[0][0]) for f in mat_data['file_list']]
        self.labels = [int(l[0]) - 1 for l in mat_data['labels']]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}. Check path case sensitivity.")
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    normalize
])

print("[INFO] Setting up PyTorch datasets...")
train_dataset = StanfordDogsDataset(TRAIN_MAT, IMAGES_DIR, transform=train_transform)
test_dataset = StanfordDogsDataset(TEST_MAT, IMAGES_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

NUM_CLASSES = len(np.unique(train_dataset.labels))
print(f"[INFO] Total Classes: {NUM_CLASSES}")

# =====================================================================
# 2. CUSTOM MODEL WRAPPERS (Preserves Grad-CAM Access paths)
# =====================================================================
class DogBreedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class DogBreedResNet(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedResNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class DogBreedMobileNet(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedMobileNet, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_model(model_name):
    if model_name == "ResNet50":
        return DogBreedResNet(NUM_CLASSES).to(device)
    elif model_name == "EfficientNetB0":
        return DogBreedEfficientNet(NUM_CLASSES).to(device)
    elif model_name == "MobileNetV3":
        return DogBreedMobileNet(NUM_CLASSES).to(device)

# =====================================================================
# 3. TRAINING ENGINE LOOP 
# =====================================================================
model_names = ["MobileNetV3", "ResNet50", "EfficientNetB0"]
val_acc_histories = {}

plt.figure(figsize=(10, 6))

for name in model_names:
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    model = get_model(name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets.data)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == targets.data)
                
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct.double() / len(test_loader.dataset)
        
        history.append(epoch_val_acc.item())
        scheduler.step(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"best_{name}.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[INFO] Early stopping triggered for {name} at epoch {epoch+1}")
                break
                
    val_acc_histories[name] = history
    plt.plot(history, label=f"{name} Val Acc")

plt.title("Stanford Dogs - PyTorch Performance Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("pytorch_model_comparison.png")
print("\n[INFO] Comparison chart saved as 'pytorch_model_comparison.png'")

# =====================================================================
# 4. EXPLAINABLE AI CONTRIBUTION (GRAD-CAM FIXED FOR LINUX)
# =====================================================================
print("\n" + "="*60)
print("[PROPOSAL CONTRIBUTION] Generating PyTorch Grad-CAM...")
print("="*60)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Only register the forward hook
        self.target_layer.register_forward_hook(self.save_features)
        
    def save_features(self, module, input, output):
        self.features = output
        # Attach the gradient hook directly to the output tensor!
        output.register_hook(self.save_gradients)
        
    def save_gradients(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Calculate Grad-CAM heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        features = self.features.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        
        # Safety check to avoid division by zero
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

try:
    eval_model = get_model("EfficientNetB0")
    eval_model.load_state_dict(torch.load("best_EfficientNetB0.pth", map_location=device))
    eval_model.eval()
    
    # Unfreeze the model so Grad-CAM can calculate the heatmap gradients
    for param in eval_model.parameters():
        param.requires_grad = True
        
    target_layer_block = eval_model.backbone.features[-1]
    cam_extractor = GradCAM(eval_model, target_layer_block)
    
    # Pick 5 random images from the test set
    num_images = 5
    random_indices = np.random.choice(len(test_dataset), num_images, replace=False)
    
    # Create a tall figure (5 rows, 2 columns)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))
    
    for i, idx in enumerate(random_indices):
        sample_tensor, _ = test_dataset[idx]
        
        # Tell PyTorch to track gradients for the input image
        input_tensor = sample_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True) 
        
        heatmap = cam_extractor.generate(input_tensor)
        
        inv_img = sample_tensor.permute(1, 2, 0).numpy()
        inv_img = inv_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        inv_img = np.clip(inv_img, 0, 1)
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        superimposed = heatmap_colored * 0.4 + inv_img
        superimposed = np.clip(superimposed, 0, 1)
        
        # Plot Original
        axes[i, 0].imshow(inv_img)
        axes[i, 0].set_title(f"Original Dog (Test Index {idx})")
        axes[i, 0].axis("off")
        
        # Plot Grad-CAM
        axes[i, 1].imshow(superimposed)
        axes[i, 1].set_title("Grad-CAM Focus")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    plt.savefig("pytorch_gradcam_5_results.png")
    print(f"[SUCCESS] Saved 5 Grad-CAM analysis charts to 'pytorch_gradcam_5_results.png'")

except Exception as e:
    print(f"[WARNING] Could not extract Grad-CAM metrics directly: {str(e)}")

print("\n[INFO] PyTorch script finished execution perfectly.")

