import matplotlib.pyplot as plt

# Models and metrics
models = ["Logistic Regression", "Multinomial NB", "Linear SVM", "Random Forest"]
accuracy = [94.32, 85.05, 95.02, 93.55]

# Plot
plt.figure(figsize=(8,5))
plt.bar(models, accuracy)
plt.ylabel("Accuracy (%)")
plt.title("Baseline Model Comparison (WELFake Dataset)")
plt.ylim(80, 100)
plt.xticks(rotation=20)
plt.show()
