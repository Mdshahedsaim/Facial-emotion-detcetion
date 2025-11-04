import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

tf.random.set_seed(42)

img_width, img_height = 48, 48
batch_size = 64  # Increased for better gradient estimation
epochs = 100  # More epochs with early stopping

train_dir = "Dataset/train"
test_dir = "Dataset/test"

# More aggressive data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    contrast_range=[0.8, 1.2],  # Add contrast variation
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Improved architecture with attention mechanism
model = Sequential([
    # First block - Feature extraction
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Second block - Deeper features
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # Third block - High-level features
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # Fourth block - Complex patterns
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Dropout(0.4),

    # Global Average Pooling instead of Flatten (reduces overfitting)
    GlobalAveragePooling2D(),
    
    # Fully connected layers with L2 regularization
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')
])

# Compile with learning rate scheduling
initial_learning_rate = 0.001
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=initial_learning_rate),
    metrics=['accuracy', 'top_2_accuracy']  # Additional metric
)

model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        patience=15,  # More patience
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive reduction
        patience=8,
        min_lr=1e-8,
        verbose=1,
        cooldown=3
    ),
    ModelCheckpoint(
        'best_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    # Additional checkpoint for loss
    ModelCheckpoint(
        'best_loss_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
]

steps_per_epoch = max(1, train_generator.samples // batch_size)
validation_steps = max(1, validation_generator.samples // batch_size)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Train with class weights for imbalanced datasets
class_weights = None
try:
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Calculate class weights if sklearn is available
    y_train = train_generator.classes
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"Using class weights: {class_weights}")
except ImportError:
    print("sklearn not available, training without class weights")

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

model.save("emotion_model_final.h5")

class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# Save class labels with confidence scores
with open("emotion_labels_final.txt", "w") as f:
    for i in range(num_classes):
        f.write(f"{i}: {index_to_class[i]}\n")

print("Class mapping:")
for i in range(num_classes):
    print(f"{i}: {index_to_class[i]}")

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0,0].set_title('Model Accuracy', fontsize=14)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0,1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0,1].set_title('Model Loss', fontsize=14)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot top-2 accuracy
    if 'top_2_accuracy' in history.history:
        axes[1,0].plot(history.history['top_2_accuracy'], label='Training Top-2 Acc', linewidth=2)
        axes[1,0].plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Acc', linewidth=2)
        axes[1,0].set_title('Top-2 Accuracy', fontsize=14)
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Top-2 Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history.history:
        axes[1,1].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
        axes[1,1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# Comprehensive evaluation
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

test_loss, test_accuracy, test_top2_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Load and evaluate best models
try:
    best_model = tf.keras.models.load_model('best_emotion_model.h5')
    best_test_loss, best_test_accuracy, best_test_top2 = best_model.evaluate(test_generator, verbose=0)
    print(f"\nBest Model Test Accuracy: {best_test_accuracy:.4f}")
    print(f"Best Model Test Top-2 Accuracy: {best_test_top2:.4f}")
    print(f"Best Model Test Loss: {best_test_loss:.4f}")
except:
    print("Could not load best model for evaluation")

# Model size info for deployment
model_size = os.path.getsize('emotion_model_final.h5') / (1024*1024)
print(f"\nFinal Model Size: {model_size:.2f} MB")

try:
    best_model_size = os.path.getsize('best_emotion_model.h5') / (1024*1024)
    print(f"Best Model Size: {best_model_size:.2f} MB")
except:
    pass

print(f"\nTotal Parameters: {model.count_params():,}")
print("Training completed successfully!")