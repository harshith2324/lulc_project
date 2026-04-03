import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load UC Merced results
resnet = pd.read_csv('./results/ucmerced/resnet50_ucmerced.csv')
vit = pd.read_csv('./results/ucmerced/vit_ucmerced.csv')
swin = pd.read_csv('./results/ucmerced/swin_ucmerced.csv')

# Combined CSV
combined = pd.concat([resnet, vit, swin], ignore_index=True)
combined.to_csv('./results/ucmerced/all_ucmerced_results.csv', index=False)
print("Combined CSV saved!")
print(combined.to_string(index=False))

models = combined['Model'].tolist()
accuracy = combined['Accuracy'].tolist()
precision = combined['Precision'].tolist()
recall = combined['Recall'].tolist()
f1 = combined['F1'].tolist()
comp_time = combined['Computation_Time_Seconds'].tolist()

x = np.arange(len(models))
width = 0.2

# Graph 1: Performance metrics
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
ax.bar(x - 0.5*width, precision, width, label='Precision')
ax.bar(x + 0.5*width, recall, width, label='Recall')
ax.bar(x + 1.5*width, f1, width, label='F1 Score')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison - UC Merced Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./results/ucmerced/performance_comparison_ucmerced.png', dpi=150)
print("Performance graph saved.")

# Graph 2: Computation Time
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, comp_time, color=['steelblue', 'darkorange', 'green'])
ax.set_xlabel('Model')
ax.set_ylabel('Computation Time (seconds)')
ax.set_title('Inference Computation Time by Model - UC Merced Dataset')
ax.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars, comp_time):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('./results/ucmerced/computation_time_ucmerced.png', dpi=150)
print("Computation time graph saved.")

# Graph 3: EuroSAT vs UC Merced Accuracy Comparison
fig, ax = plt.subplots(figsize=(10, 6))
eurosat_acc = [0.90, 0.65, 0.75]  # ResNet, ViT, Swin
ucmerced_acc = accuracy

x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, eurosat_acc, width, label='EuroSAT', color='steelblue')
ax.bar(x + width/2, ucmerced_acc, width, label='UC Merced', color='darkorange')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison: EuroSAT vs UC Merced')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./results/ucmerced/eurosat_vs_ucmerced_accuracy.png', dpi=150)
print("Comparison graph saved.")

plt.show()
print("\nAll done!")