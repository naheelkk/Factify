import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Logistic Regression", "Multinomial NB", "Linear SVM", "Random Forest"]

# Metrics
accuracy = [94.32, 85.05, 95.01, 93.55]
precision = [94.5, 85.0, 95.0, 93.0]  # Approx averages from your report
recall = [94.5, 85.5, 95.0, 93.5]
f1 = [94.5, 85.0, 95.0, 93.5]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, accuracy, width, label="Accuracy")
rects2 = ax.bar(x - 0.5*width, precision, width, label="Precision")
rects3 = ax.bar(x + 0.5*width, recall, width, label="Recall")
rects4 = ax.bar(x + 1.5*width, f1, width, label="F1-Score")

# Labels & formatting
ax.set_ylabel("Percentage (%)")
ax.set_title("Baseline Model Performance on WELFake Dataset")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.legend()

# Add value labels
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()
