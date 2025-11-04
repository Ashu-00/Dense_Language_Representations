import matplotlib.pyplot as plt
import json
import numpy as np

with open('bias_results.json', 'r') as f:
    bias_results = json.load(f)

models = list(bias_results.keys())
categories = list(bias_results[models[0]].keys())

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.8 / len(models)  
r = np.arange(len(categories))

for i, model in enumerate(models):
    offset = i * bar_width
    ax.bar(r + offset, [bias_results[model][cat] for cat in categories], 
           width=bar_width, label=model)

ax.set_xlabel('Bias Categories', fontweight='bold')
ax.set_ylabel('Bias Score', fontweight='bold')
ax.set_title('Bias Comparison Across Models', fontweight='bold')
ax.set_xticks(r + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()


ax.text(-0.1, 1.0, 'Ideal = 0', transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='center')

plt.grid()
plt.tight_layout()
plt.savefig('bias_comparison.png', dpi=300)