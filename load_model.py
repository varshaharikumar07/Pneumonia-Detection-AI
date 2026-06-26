import tensorflow as tf

model = tf.keras.models.load_model("pneumonia_model.keras")
model.summary()
