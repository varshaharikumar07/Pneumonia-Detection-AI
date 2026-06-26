import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("pneumonia_model.keras")

IMG_SIZE = (150, 150)

def predict_xray(img_path):
    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize (VERY IMPORTANT)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Print raw score
    print("Raw confidence score:", prediction)

    # Decision
    if prediction >= 0.5:
        print("🦠 Result: PNEUMONIA")
    else:
        print("🫁 Result: NORMAL")


img_path = "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

predict_xray(img_path)