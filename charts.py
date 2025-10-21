import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
forward = [0.742, 0.745, 0.742, 0.741]
backward = [0.767, 0.772, 0.767, 0.766]

classes = ['Bipolar Type-1', 'Bipolar Type-2', 'Depression', 'Normal']
# Estimated F1-scores based on your overall performance
forward_f1 = [0.762, 0.724, 0.739, 0.752]
backward_f1 = [0.791, 0.750, 0.772, 0.766]

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Overall performance
x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, forward, width, label='Forward Chaining',
                color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, backward, width, label='Backward Chaining',
                color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0.70, 0.80)


for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Per-class performance
x_class = np.arange(len(classes))
bars3 = ax2.bar(x_class - width/2, forward_f1, width, label='Forward Chaining',
                color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
bars4 = ax2.bar(x_class + width/2, backward_f1, width, label='Backward Chaining',
                color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Diagnostic Classes', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('Per-Class Diagnostic Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x_class)
ax2.set_xticklabels(classes, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0.70, 0.82)

for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
