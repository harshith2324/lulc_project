import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  results
resnet = pd.read_csv('./results/resnet50_results.csv')
vit = pd.read_csv('./results/vit_results.csv')
swin = pd.read_csv('./results/swin_results.csv')

# Combined CSV
combined = pd.concat([resnet, vit, swin], ignore_index=True)
combined.to_csv('./results/all_results.csv', index=False)
print("Combined CSV saved to results/all_results.csv")
print(combined.to_string(index=False))

models = combined['Model'].tolist()
accuracy = combined['Accuracy'].tolist()
precision = combined['Precision'].tolist()
recall = combined['Recall'].tolist()
f1 = combined['F1'].tolist()
comp_time = combined['Computation_Time_Seconds'].tolist()

x = np.arange(len(models))
width = 0.2

# Graph 1: Accuracy, Precision, Recall, F1
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
ax.bar(x - 0.5*width, precision, width, label='Precision')
ax.bar(x + 0.5*width, recall, width, label='Recall')
ax.bar(x + 1.5*width, f1, width, label='F1 Score')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./results/performance_comparison.png', dpi=150)
print("Performance graph saved.")

# Graph 2: Computation Time
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, comp_time, color=['steelblue', 'darkorange', 'green'])
ax.set_xlabel('Model')
ax.set_ylabel('Computation Time (seconds)')
ax.set_title('Inference Computation Time by Model')
ax.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars, comp_time):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('./results/computation_time.png', dpi=150)
print("Computation time graph saved.")

plt.show()
print("\nAll done!")
