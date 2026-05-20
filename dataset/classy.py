import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("Please wait.")

 
data_dir =r"C:\Users\Felishiya\Desktop\dataset"

 
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

 
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

 
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(train_ds.class_names), activation='softmax') 
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 
print("\nStarting training... Look at the numbers go!")
model.fit(train_ds, validation_data=val_ds, epochs=3)

print("\n PROJECT FINISHED! ")
print("Celebrity names the AI successfully learned:", train_ds.class_names)

import numpy as np

print("\n--- FINAL TEST: PREDICTING A NEW IMAGE ---")

 
test_image_path = r"C:\Users\Felishiya\Desktop\test.jpg"
 
img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

 
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

 
class_names = train_ds.class_names
winner = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"\n AI PREDICTION: I am {confidence:.2f}% sure this is a picture of {winner}!")