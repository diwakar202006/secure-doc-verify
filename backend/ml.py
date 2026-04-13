import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess(image):
    # ✅ Ensure image is RGB (fix RGBA / grayscale issue)
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Normalize
    image = np.array(image) / 255.0

    # Expand dimensions (batch size = 1)
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


def predict_image(path):
    try:
        print("📂 Loading image:", path)

        # Open image safely
        image = Image.open(path).convert("RGB")

        # Preprocess
        img = preprocess(image)
        print("📏 Input shape:", img.shape)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        print("🧠 Raw output:", output)

        prediction = output[0][0]

        return {
            "result": "fake" if prediction > 0.5 else "real",
            "confidence": float(prediction)
        }

    except Exception as e:
        print("❌ ML Error:", str(e))
        return {
            "error": str(e)
        }