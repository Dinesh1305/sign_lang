import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# STEP 1: DATASET SETUP
# -----------------------------
DATASET_PATH = "dataset/"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

train_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# STEP 2: MODEL ARCHITECTURE
# -----------------------------
model = Sequential([
    # Layer 1
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),

    # Layer 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Dropout(0.25),

    # Layer 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Dropout(0.25),

    # Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.30),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# -----------------------------
# STEP 3: TRAIN WITH EARLY STOPPING
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop]
)

# -----------------------------
# STEP 4: SAVE MODEL
# -----------------------------
model.save("hello_detector.keras")  # modern save format
print("\n🎉 Training completed. Saved as hello_detector.keras")
print("🏷️ Label mapping:", train_ds.class_indices)
