import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

from data_loder import train_generator, val_generator, class_weight_dict
from model_builder import build_enhanced_model  
import matplotlib.pyplot as plt
from datetime import datetime
import os

save_dir = os.path.join('saved_models', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir, exist_ok=True)

model = build_enhanced_model()
print("Model built successfully. Summary:")
model.summary()  

from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    CSVLogger
)

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(save_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    CSVLogger(os.path.join(save_dir, 'training_log.csv'))
]

print("🚀 Starting training...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight_dict,
)

print(f"✅ Training complete. Model saved to {save_dir}")

model.save(os.path.join(save_dir, 'final_model.h5'))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
plt.show()

from sklearn.metrics import classification_report
import numpy as np

print("🧪 Evaluating on validation set...")
y_true = val_generator.classes
y_pred = model.predict(val_generator, batch_size=32, verbose=1)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))