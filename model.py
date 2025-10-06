import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------
# Data Generators
# ------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=50
)

test_datagen = ImageDataGenerator(rescale=1./255)

# NOTE: For multi-task learning, your dataset should include:
#   - class labels (Dent, Scratch, etc.)
#   - damage percentage (0-100) in a CSV or numpy array
# Example assumes you already have these aligned.

training_set = train_datagen.flow_from_directory(
    r"F:\Project\Dataset\training",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

testing_set = test_datagen.flow_from_directory(
    r"F:\Project\Dataset\testing",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ------------------------
# Model Architecture
# ------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output 1: Damage Type (classification)
damage_type = Dense(len(training_set.class_indices), activation='softmax', name="damage_type")(x)

# Output 2: Damage Percentage (regression)
damage_percent = Dense(1, activation='linear', name="damage_percent")(x)

# Combine into multi-output model
model = Model(inputs=base_model.input, outputs=[damage_type, damage_percent])

# ------------------------
# Compile Model
# ------------------------
model.compile(
    optimizer='adam',
    loss={
        "damage_type": "categorical_crossentropy",
        "damage_percent": "mse"   # mean squared error for regression
    },
    metrics={
        "damage_type": "accuracy",
        "damage_percent": "mae"   # mean absolute error
    }
)

model.summary()

# ------------------------
# Training (Example)
# ------------------------
checkpoint = ModelCheckpoint(
    'best_multitask_model.weights.h5',
    monitor='val_damage_type_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=True
)

# NOTE: For multi-output, you need to provide both y_class and y_reg values.
# Example: model.fit(x_train, {"damage_type": y_classes, "damage_percent": y_percentages}, ...)

# Placeholder fit (replace with your dataset that includes damage % labels)
# history = model.fit(training_images,
#                     {"damage_type": y_classes, "damage_percent": y_percentages},
#                     epochs=50,
#                     validation_data=(val_images, {"damage_type": y_val_classes, "damage_percent": y_val_percent}),
#                     callbacks=[checkpoint])

# ------------------------
# Prediction Example
# ------------------------
from tensorflow.keras.preprocessing import image

img_path = r"C:\Users\Tushar\Downloads\car.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred_type, pred_percent = model.predict(img_array)

damage_classes = list(training_set.class_indices.keys())
predicted_class = damage_classes[np.argmax(pred_type)]
predicted_percentage = float(pred_percent[0][0])

print(f"Predicted Damage Type: {predicted_class}")
print(f"Estimated Damage Percentage: {predicted_percentage:.2f}%")
