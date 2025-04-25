import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # ✅ Progress bar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ==== SETTINGS ====
IMG_SIZE = 64
CSV_PATH = "E:/KEERTHANA NNML/HAM10000_metadata.csv"
PART1_DIR = "E:/KEERTHANA NNML/HAM10000_images_part_1"
PART2_DIR = "E:/KEERTHANA NNML/HAM10000_images_part_2"
MODEL_PATH = "E:/KEERTHANA NNML/ham10000_model.h5"

# ==== 1. Load metadata ====
print("Loading metadata...")
df = pd.read_csv(CSV_PATH)

# ==== 2. Map image_id to full paths ====
print("Mapping image paths...")
image_paths = {}
for folder in [PART1_DIR, PART2_DIR]:
    for file in os.listdir(folder):
        if file.endswith('.jpg'):
            image_id = os.path.splitext(file)[0]
            image_paths[image_id] = os.path.join(folder, file)

df['path'] = df['image_id'].map(image_paths)

# Drop rows with missing images
df = df[df['path'].notna()]



# ==== 3. Preprocess images with simple progress percentage ====
print("Preprocessing images...")
images = []
labels = []
total = len(df)

for idx, row in enumerate(df.itertuples(), 1):
    path = row.path
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(row.dx)

    # Show percentage every 10%
    if idx % (total // 10) == 0 or idx == total:
        percent = int((idx / total) * 100)
        print(f"Preprocessing images: {percent}% done")

# ✅ Convert and encode after loop
X = np.array(images)

le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y = to_categorical(y_encoded)

# ✅ Save label map
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
reverse_label_map = {v: k for k, v in label_map.items()}
print("Label Map:", label_map)


# ==== 4. Train/Test Split (with stratify) ====
print("Splitting dataset with stratification...")
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded)

y_test = np.argmax(y_test_cat, axis=1)

# ==== 5. Load the trained model ====
print("Loading model...")
model = load_model(MODEL_PATH)

# ==== 6. Predict ====
print("Making predictions...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==== 7. Evaluation ====
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ==== 8. Confusion Matrix ====
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
