import tensorflow as tf

# Load model WITHOUT compiling (IMPORTANT FIX)
model = tf.keras.models.load_model("model.h5", compile=False)

# Create concrete function (NEW SAFE METHOD)
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 224, 224, 3], tf.float32)
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# 🔥 Compatibility fix
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# Save model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ FINAL TFLite model created successfully!")