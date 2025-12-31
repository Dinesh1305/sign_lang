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

    # 🌟 NEW settings for higher accuracy
    rotation_range=35,
    zoom_range=0.4,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.3, 1.6],
    channel_shift_range=50.0,
    horizontal_flip=True,
    shear_range=0.2
)


train_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # MULTI-CLASS
    subset='training'
)

val_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# STEP 2: MODEL FOR 10 CLASSES
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(train_ds.num_classes, activation='softmax')  # 10 CLASSES
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# STEP 3: TRAIN WITH EARLY STOPPING
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# -----------------------------
# STEP 4: SAVE MODEL
# -----------------------------
model.save("digit_detector.keras")
print("\n🎉 Training complete! Model saved as digit_detector.keras")
print("🏷️ Label mapping:", train_ds.class_indices)
