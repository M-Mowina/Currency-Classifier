from tensorflow import keras
from keras.models import Sequential, save_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

train_data_generation = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           )

training_set = train_data_generation.flow_from_directory('dataset\\train',
                                                         target_size=(256, 256),
                                                         batch_size=16,
                                                         class_mode='categorical')

test_data_generation = ImageDataGenerator(rescale=1./255)
testing_set = test_data_generation.flow_from_directory('dataset\\test',
                                                       target_size=(256, 256),
                                                       batch_size=16,
                                                       class_mode='categorical')

valid_data_generation = ImageDataGenerator(rescale=1./255)
validation_set = valid_data_generation.flow_from_directory('dataset\\valid',
                                                       target_size=(256, 256),
                                                       batch_size=16,
                                                       class_mode='categorical')

input_shape = (256, 256, 3)
model = Sequential([
    # Convolutional layers for feature extraction
    Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),


    # Flatten for dense layers
    Flatten(),

    # Dense layers for classification
    Dense(256, activation='relu'),
    Dropout(0.2),  # Optional dropout to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),  # Optional dropout
    Dense(9, activation='softmax'),  # Output layer with 6 units for 6 currencies
])

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x = training_set, validation_data = validation_set, epochs=40,batch_size=32)

model.save('model4.h5')