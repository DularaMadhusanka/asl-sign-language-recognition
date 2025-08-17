import os
import hashlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input  # Updated for EfficientNet
from sklearn.utils.class_weight import compute_class_weight
import json
from collections import Counter

def remove_duplicates_from_folder(folder_path):
    hashes = set()
    for root, _, files in os.walk(folder_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in hashes:
                os.remove(fpath)
                print(f"Removed duplicate: {fpath}")
            else:
                hashes.add(filehash)

remove_duplicates_from_folder('ASL_Alphabet_Dataset/asl_alphabet_train')
remove_duplicates_from_folder('ASL_Alphabet_Dataset/asl_alphabet_test')

for unwanted in ['del', 'space', 'nothing']:
    train_path = os.path.join('ASL_Alphabet_Dataset/asl_alphabet_train', unwanted)
    test_path = os.path.join('ASL_Alphabet_Dataset/asl_alphabet_test', unwanted)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
        print(f"Removed class folder: {train_path}")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        print(f"Removed class folder: {test_path}")

train_dir = 'ASL_Alphabet_Dataset/asl_alphabet_train'
test_dir = 'ASL_Alphabet_Dataset/asl_alphabet_test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2 
SEED = 42
NUM_WORKERS = 4

# Data augmentation 
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights using generator classes
classes = np.unique(train_generator.classes)
weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

# Save class indices mapping for inference/serving
class_map_path = os.path.join(os.getcwd(), 'class_indices.json')
with open(class_map_path, 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

def get_generators():
    return train_generator, val_generator, test_generator, class_weight_dict

# Class distribution analysis (using generator counts)
inv_map = {v: k for k, v in train_generator.class_indices.items()}
counter = Counter(train_generator.classes)
sorted_indices = sorted(counter.keys())
class_names = [inv_map[i] for i in sorted_indices]
class_counts = [counter[i] for i in sorted_indices]

plt.figure(figsize=(15, 6))
bars = plt.bar(class_names, class_counts)
plt.title('Class Distribution in Training Data', pad=20)
plt.xlabel('ASL Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Helper to reverse EfficientNet preprocessing
def deprocess_efficientnet(img):
    img = img.copy()
    img = img * 255.0
    return np.clip(img, 0, 255).astype('uint8')

# Sample images visualization
sample_images, sample_labels = next(train_generator)
plt.figure(figsize=(12, 12))
for i in range(min(9, sample_images.shape[0])):
    plt.subplot(3, 3, i+1)
    plt.imshow(deprocess_efficientnet(sample_images[i]))
    class_idx = int(np.argmax(sample_labels[i]))
    plt.title(f"{class_names[class_idx]} ({class_idx})")
    plt.axis('off')
plt.suptitle('Augmented Training Samples', y=1.02)
plt.tight_layout()
plt.show()

print(f"\n🔍 Dataset Summary:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Number of classes: {len(class_names)}")
print("Class weights for imbalance:", class_weight_dict)