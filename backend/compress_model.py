import tensorflow as tf

model = tf.keras.models.load_model("model.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔥 Keep it simple (NO experimental flags)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Simple TFLite model created")