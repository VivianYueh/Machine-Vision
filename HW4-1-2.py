import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetV2S

def CCSN_images(dataset_dir):
    classfolder = glob.glob(os.path.join(dataset_dir,"*"),recursive=True)
    class_name  = [f.split(os.path.sep)[-1] for f in classfolder]
    img_labels  = []
    img_list    = []

    for class_id, f in enumerate(classfolder):
        files = glob.glob(os.path.join(f,"*.jpg"),recursive=True)
        img_labels.extend([class_id]*len(files))
        img_list.extend(files)

    img_labels = np.array(img_labels)
    img_list   = np.array(img_list,dtype=object)
    
    return img_list.reshape((-1,1)), img_labels, class_name

CCSNDataset_Path = "D:\大學\大三\機器視覺\HW4\database\Kaggle\CCSN\CCSN_v2"
img_list, img_labels, label_names = CCSN_images(CCSNDataset_Path)

# 转换为 tf.data.Dataset
def load_dataset(img_list, img_labels, batch_size, img_size=(224, 224)):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, img_labels))

    def process_path(file_path, label):
        img = tf.io.read_file(file_path[0])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0  # Normalize to [0,1]
        return img, label

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(img_labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 64
dataset = load_dataset(img_list, img_labels, batch_size)

from tensorflow.keras import layers, models, applications

def build_efficient(class_number):
    base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # 决定是否冻结预训练模型

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(class_number, activation='softmax')
    ])

    return model

class_number = len(label_names)
model = build_efficient(class_number)
model.summary()

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(img_list, img_labels, test_size=0.2, random_state=42)

train_dataset = load_dataset(X_train, y_train, batch_size)
valid_dataset = load_dataset(X_valid, y_valid, batch_size)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 100
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
