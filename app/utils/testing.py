from PIL import Image
import keras.models
import numpy as np 

# Load your trained model
model = keras.models.load_model('my_model.h5')

def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Resize to 256x256 pixels
    img = img.resize((256, 256))

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to a numpy array
    img_array = np.array(img)

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    # Expand dimensions to create a single-image batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Get the image path from user input or another source
image_path = 'dataset\\train\\200\\200.2.jpg'

# Preprocess the image
preprocessed_img = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_img)

# Decode prediction (assuming categorical labels)
predicted_class = np.argmax(prediction)
print(f"Predicted label: {predicted_class} prediction: {prediction}")