import matplotlib.pyplot as plt

# Data for F1 vs Alpha for KD models with teacher en fr audio on test set
alphas = [0.5, 0.7, 0.8, 0.9]
f1_scores = [0.6102941222104997, 0.6101713163784356, 0.6066712583020145, 0.6097224186641287]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(alphas, f1_scores, marker='o', linestyle='-', color='b', label='F1 Macro')
plt.xlabel('Alpha')
plt.ylabel('F1 Macro Score')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('results/f1_vs_alpha_en_fr_audio.pdf')
print("Graph saved as results/f1_vs_alpha_en_fr_audio.pdf")