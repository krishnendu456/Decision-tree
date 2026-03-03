# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create model
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# 4. Train model
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 8. Plot Decision Tree
plt.figure(figsize=(15,8))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)
plt.title("Decision Tree (Breast Cancer Dataset)")
plt.show()

# 9. Accuracy Plot (Single Bar)
plt.bar(["Decision Tree"], [accuracy])
plt.ylim(0.8, 1.0)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.show()