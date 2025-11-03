import matplotlib.pyplot as plt
import json
import numpy as np

with open('bias_results.json', 'r') as f:
    bias_results = json.load(f)

models = list(bias_results.keys())
categories = list(bias_results[models[0]].keys())

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]

bars1 = ax.bar(r1, [bias_results[models[0]][cat] for cat in categories], 
               width=bar_width, label=models[0], color='skyblue')
bars2 = ax.bar(r2, [bias_results[models[1]][cat] for cat in categories], 
               width=bar_width, label=models[1], color='salmon')

ax.set_xlabel('Bias Categories', fontweight='bold')
ax.set_ylabel('Bias Score', fontweight='bold')
ax.set_title('Bias Comparison Between DistilBERT and BERT Models', fontweight='bold')
ax.set_xticks([r + bar_width/2 for r in range(len(categories))])
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()


ax.text(-0.1, 1.0, 'Ideal = 0', transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='center')

plt.grid()
plt.tight_layout()
plt.savefig('bias_comparison.png', dpi=300)