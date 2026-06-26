# evaluate_thresholds.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

model = tf.keras.models.load_model("pneumonia_model.keras")

test_dir = r"C:\Users\Varsha\Desktop\Pneumonia Proj\dataset\chest_xray\test"

img_size = (224, 224)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds).ravel()

thresholds = [0.5, 0.6, 0.7, 0.8]
for t in thresholds:
    y_pred = (y_pred_probs > t).astype(int)
    accuracy = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\nThreshold {t}:")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 Score = {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
