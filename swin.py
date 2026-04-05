import time
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchgeo.models import swin_v2_t, Swin_V2_T_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

#  setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# EuroSAT band order: B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12
# Swin needs: B04,B03,B02,B05,B06,B07,B08,B11,B12
# Corresponding indices in EuroSAT:
BAND_INDICES = [3, 2, 1, 4, 5, 6, 7, 11, 12]

# Dataset
print("Loading EuroSAT dataset...")
train_dataset = EuroSAT(root="./data", split="train", download=True)
test_dataset = EuroSAT(root="./data", split="test", download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Sentinel-2 weights
print("Loading Swin-T with Sentinel-2 pretrained weights...")
weights = Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS
model = swin_v2_t(weights=weights)
model.head = nn.Linear(model.head.in_features, 10)
model = model.to(device)

# Fine tune
print("Fine-tuning model on EuroSAT training data...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        images = batch['image'].float().to(device)
        labels = batch['label'].to(device)

        # Select the 9 required bands
        images = images[:, BAND_INDICES, :, :]

        # Resize to 256x256 for Swin-T
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# Evaluating
print("\nRunning inference on test set...")
model.eval()
all_preds = []
all_labels = []

start_time = time.time()

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].float().to(device)
        labels = batch['label'].to(device)

        # Select the 9 required bands
        images = images[:, BAND_INDICES, :, :]

        # Resize to 256x256 for Swin-T
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

end_time = time.time()
computation_time = end_time - start_time

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Computation Time (testing only): {computation_time:.2f} seconds")

# CSV
results = pd.DataFrame([{
    'Model': 'Swin-T',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'Computation_Time_Seconds': computation_time
}])

results.to_csv('./results/swin_results.csv', index=False)
print("\nResults saved to results/swin_results.csv")
