import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# ---------------------------
# 1. Configuration
# ---------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

IMAGE_DIR = 'img_align_celeba'
IDENTITY_FILE = 'identity_CelebA.txt'

# ---------------------------
# 2. Load and Fix Identity File
# ---------------------------
# Make sure the identity file has two columns: image_id and identity
df = pd.read_csv(IDENTITY_FILE, delim_whitespace=True, header=None, names=['image_id', 'identity'])

# ---------------------------
# 3. Reduce to 100 Identities
# ---------------------------
# Select only 100 unique identities for faster training
selected_ids = df['identity'].unique()[:100]
df = df[df['identity'].isin(selected_ids)].reset_index(drop=True)

num_classes = df['identity'].nunique()
print(f"Training with {num_classes} identities")

# ---------------------------
# 4. Train/Val/Test Split
# ---------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df_train = df[:int(0.8 * len(df))]
df_val = df[int(0.8 * len(df)):int(0.9 * len(df))]
df_test = df[int(0.9 * len(df)):]

# ---------------------------
# 5. Image Generators
# ---------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    df_train, IMAGE_DIR, x_col='image_id', y_col='identity',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='sparse'
)

val_generator = val_datagen.flow_from_dataframe(
    df_val, IMAGE_DIR, x_col='image_id', y_col='identity',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='sparse'
)

test_generator = test_datagen.flow_from_dataframe(
    df_test, IMAGE_DIR, x_col='image_id', y_col='identity',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='sparse'
)

# ---------------------------
# 6. Build CNN Model
# ---------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# ---------------------------
# 7. Compile and Train
# ---------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# ---------------------------
# 8. Evaluate
# ---------------------------
loss, acc = model.evaluate(test_generator)
print("Test Accuracy:", acc)

# ---------------------------
# 9. Predict Single Image
# ---------------------------
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction)

# Example:
sample_img_path = os.path.join(IMAGE_DIR, df_test.iloc[0]['image_id'])
predicted_id = predict_image(model, sample_img_path)
print(f"Predicted Identity Class: {predicted_id}")

