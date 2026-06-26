# test_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("pneumonia_model_finetuned.keras")

# Path to a sample X-ray image
img_path = r"C:\Users\Varsha\Desktop\chn.jpg"   # <-- replace with your image file path

# Preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run prediction
prediction = model.predict(img_array)[0][0]
print("Pneumonia probability:", prediction)

# Adjusted threshold decision
if prediction > 0.7:   # <-- you can change 0.7 to 0.6, 0.75, etc.
    print("🦠 Pneumonia detected")
else:
    print("🫁 Normal")

