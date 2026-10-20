from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Jumlah data: {len(data.data)}")
print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=data.target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
