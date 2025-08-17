# 💡 Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Load Dataset
iris = load_iris()
X = iris.data         # Features
y = iris.target       # Labels

print("🌼 Feature names:", iris.feature_names)
print("🌼 Target classes:", iris.target_names)

# ✂️ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧼 Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🌲 Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# 🔮 Predict on test data
y_pred = model.predict(X_test_scaled)

# ✅ Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", round(accuracy * 100, 2), "%")

# 🧾 Classification Report & Confusion Matrix
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🧠 Bonus: Built-in Function Use
correct = sum(y_pred == y_test)
total = len(y_test)
print(f"🎯 Correct predictions: {correct} / {total}")
print(f"🎯 Accuracy: {correct / total * 100:.2f}%")