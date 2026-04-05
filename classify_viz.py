import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights

CLASS_NAMES = [
    'Annual Crop', 'Forest', 'Herbaceous Vegetation',
    'Highway', 'Industrial', 'Pasture',
    'Permanent Crop', 'Residential', 'River', 'Sea/Lake'
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_dataset = EuroSAT(root="./data", split="test", download=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0)

weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataset = EuroSAT(root="./data", split="train", download=True)
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
images_tensor = batch['image'].float().to(device)
labels = batch['label']

with torch.no_grad():
    outputs = model(images_tensor)
    preds = torch.argmax(outputs, dim=1).cpu()

# Using torchGeo's built-in plot() method for each sample
print("\nGenerating TorchGeo batch visualizations...")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('ResNet-50 Classification Results (TorchGeo Visualization)', 
             fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    sample = test_dataset[i]
    # Use TorchGeo's built-in plot method
    sample_fig = test_dataset.plot(sample)
    sample_fig.canvas.draw()
    
    import numpy as np
    buf = sample_fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    plt.close(sample_fig)
    
    correct = preds[i].item() == labels[i].item()
    color = 'green' if correct else 'red'
    label_text = '✓ CORRECT' if correct else '✗ WRONG'
    
    ax.imshow(img_array)
    ax.set_title(
        f'{label_text}\nTrue: {CLASS_NAMES[labels[i]]}\nPred: {CLASS_NAMES[preds[i]]}',
        fontsize=8, fontweight='bold', color=color
    )
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(6)

plt.tight_layout()
plt.savefig('./results/classification_viz.png', dpi=150, bbox_inches='tight')
print("Visualization saved to results/classification_viz.png")
plt.show()
