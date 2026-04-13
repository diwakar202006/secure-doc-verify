import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional but IMPORTANT) Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# Save compressed model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite successfully!")