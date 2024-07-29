from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

model = tf.keras.models.load_model('my_weights_model.h5')

@app.route('/')
def home():
    return render_template('site.html')

@app.route("/upload", methods=["POST"])
def upload():
    image_file = request.files["image"]
    img = Image.open(image_file)  # Ensure proper conversion based on model input

    # Preprocess image based on your model requirements (e.g., resizing, normalization)
    preprocessed_img = preprocess_image(img)

    # Predict lapel using your model
    label = model.predict(preprocessed_img)
    predicted_class = np.argmax(label)
    #result = predicted_class

    if predicted_class == 8: result = '1 EGP'
    elif predicted_class == 0: result = '10 EGP'
    elif predicted_class == 1: result = '10 EGP(new)'
    elif predicted_class == 2: result = '100 EGP'
    elif predicted_class == 3: result = '20 EGP'
    elif predicted_class == 4: result = '20 EGP(new)'
    elif predicted_class == 5: result = '200 EGP'
    elif predicted_class == 6: result = '5 EGP'
    elif predicted_class == 7: result = "50 EGP"

    # Convert image to base64 for efficient transfer
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format=img.format)
    encoded_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

    print(result)          
    return render_template('results.html', result=result, image=encoded_image)


def preprocess_image(img):
    # Resize if needed
    if img.size != (256, 256):
        img = img.resize((256, 256))

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Normalize
    img_array = np.array(img) / 255.0

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if __name__ == '__main__':
    app.run()
