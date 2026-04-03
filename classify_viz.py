import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT100
from torchgeo.models import resnet50, ResNet50_Weights

CLASS_NAMES = [
    'Annual Crop', 'Forest', 'Herbaceous Vegetation',
    'Highway', 'Industrial', 'Pasture',
    'Permanent Crop', 'Residential', 'River', 'Sea/Lake'
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_dataset = EuroSAT100(root="./data", split="test", download=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0)

weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataset = EuroSAT100(root="./data", split="train", download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

print("Fine-tuning for visualization...")
for epoch in range(15):
    model.train()
    for batch in train_loader:
        images = batch['image'].float().to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/15 done")

model.eval()
batch = next(iter(test_loader))
images = batch['image'].float().to(device)
labels = batch['label']

with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('ResNet-50 Classification Results on EuroSAT', fontsize=15, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Extract and normalize RGB bands properly
    img = images[i].cpu().numpy()
    rgb = img[[3, 2, 1], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    correct = preds[i].item() == labels[i].item()
    color = '#00cc00' if correct else '#cc0000'
    label_text = '✓ CORRECT' if correct else '✗ WRONG'

    ax.imshow(rgb)

    # Thick colored border
    rect = patches.Rectangle((0, 0), 63, 63, linewidth=6,
                               edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    ax.set_title(
        f'{label_text}\nTrue: {CLASS_NAMES[labels[i]]}\nPred: {CLASS_NAMES[preds[i]]}',
        fontsize=8, fontweight='bold', color=color, pad=4
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig('./results/classification_viz.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved!")
plt.show()