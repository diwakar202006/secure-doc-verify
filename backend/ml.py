import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# 🔥 SAFE MODEL LOADING
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

    print("📁 Model path:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading error:", str(e))
    interpreter = None


# 🔥 PREPROCESS FUNCTION
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


# 🔥 PREDICTION FUNCTION
def predict_image(path):
    try:
        if interpreter is None:
            return {"error": "Model not loaded properly"}

        print("📂 Loading image:", path)

        image = Image.open(path).convert("RGB")
        img = preprocess(image)

        print("📏 Input shape:", img.shape)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        print("🧠 Raw output:", output)

        prediction = output[0][0]

        return {
            "result": "fake" if prediction > 0.5 else "real",
            "confidence": float(prediction)
        }

    except Exception as e:
        print("❌ ML runtime error:", str(e))
        return {"error": str(e)}