import os
import torch
import torch.nn as nn
import timm  # Bibliothèque pour charger EfficientNet-B7
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# === Paramètres ===
DATA_DIR = 'plant_disease_dataset'
MODEL_PATH = 'plant_disease_model.pth'
BATCH_SIZE = 16
NUM_EPOCHS = 20  # Augmentation du nombre d'époques pour une meilleure convergence
LEARNING_RATE = 0.0001  # Réduction du LR pour s’adapter à un modèle plus profond
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Prétraitements optimisés ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Adapté à EfficientNet-B7
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Chargement des données ===
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Nombre de classes ===
num_classes = len(train_dataset.classes)
print(f"Classes détectées : {train_dataset.classes}")

# === Chargement du modèle EfficientNet-B7 ===
model = timm.create_model("efficientnet_b4", pretrained=True, checkpoint_path="")

# Modification de la dernière couche pour correspondre au nombre de classes
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Envoi sur le GPU si disponible
model = model.to(DEVICE)

# === Définition de la fonction de perte et de l'optimiseur ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW pour une meilleure régularisation

# === Entraînement amélioré ===
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"📘 Epoch {epoch+1} | Training Loss: {avg_loss:.4f}")

    # === Validation améliorée ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"✅ Validation Accuracy: {val_acc:.2f}%")

# === Sauvegarde du modèle ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Modèle EfficientNet-B7 sauvegardé dans {MODEL_PATH}")
