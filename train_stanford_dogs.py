"""
Course: Introduction to Deep Learning for Computer Vision / Computer Vision
Project: Fine-Grained Dog Breed Classification using the Stanford Dogs Dataset
Author: 盧春泰
"""

import os
import cv2
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for headless/remote execution, saves figures to disk
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0, MobileNetV3Large

# Force terminal configuration clarity
print("[INFO] TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("[INFO] GPUs Available: ", len(gpus))
if len(gpus) > 0:
    print("[INFO] Running on accelerated GPU environment.")
else:
    print("[WARNING] No GPU detected. Execution will fall back to CPU and will be slow!")

# =====================================================================
# 1. PATH CONFIGURATIONS & HYPERPARAMETERS
# =====================================================================
BASE_DIR = "."  
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
TRAIN_MAT = os.path.join(BASE_DIR, "train_list.mat")
TEST_MAT = os.path.join(BASE_DIR, "test_list.mat")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # Early stopping handles the actual stopping point dynamically

# =====================================================================
# 2. PARSE DATASETS & ESTABLISH PIPELINES
# =====================================================================
def load_split_from_mat(mat_path):
    print(f"[INFO] Parsing MATLAB partition lists: {os.path.basename(mat_path)}")
    mat_data = scipy.io.loadmat(mat_path)
    file_paths = [os.path.join(IMAGES_DIR, str(f[0][0])) for f in mat_data['file_list']]
    # Convert MATLAB 1-indexed targets to Python 0-indexed vectors
    labels = [int(l[0]) - 1 for l in mat_data['labels']]
    return file_paths, labels

train_paths, train_labels = load_split_from_mat(TRAIN_MAT)
test_paths, test_labels = load_split_from_mat(TEST_MAT)
NUM_CLASSES = len(np.unique(train_labels))
print(f"[INFO] Split verification: {len(train_paths)} Train samples | {len(test_paths)} Test samples.")

def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

# Spatial augmentation block to combat intra-class overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

print("[INFO] Building optimized tf.data high-performance pipelines...")
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths)).map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================================
# 3. TRANSFER LEARNING ARCHITECTURE CONSTRUCTOR
# =====================================================================
def build_transfer_model(base_model_name):
    if base_model_name == "ResNet50":
        base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif base_model_name == "EfficientNetB0":
        base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif base_model_name == "MobileNetV3":
        base_model = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    # Freeze pre-trained feature extractor representations
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # Structural regularizer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name=base_model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# =====================================================================
# 4. TRAINING pipeline WITH REGULARIZATION CALLBACKS
# =====================================================================
model_names = ["MobileNetV3", "ResNet50", "EfficientNetB0"]
training_histories = {}

plt.figure(figsize=(12, 6))

for name in model_names:
    print(f"\n{'-'*60}\n[EXECUTION] Initializing training pipeline for: {name}\n{'-'*60}")
    model = build_transfer_model(name)
    
    # Advanced evaluation mechanics as outlined in proposal setup
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(filepath=f"best_{name}.keras", monitor='val_accuracy', save_best_only=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS, 
        callbacks=[early_stop, checkpoint, lr_scheduler]
    )
    
    training_histories[name] = history.history
    
    # Add validation history curve tracking coordinates
    plt.plot(history.history["val_accuracy"], label=f"{name} Validation Accuracy")

# Format and save comparison graph metrics to filesystem
plt.title("Stanford Dogs - Model Architecture Accuracy Performance")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("model_comparison_chart.png")
print("\n[INFO] Comparison performance tracking chart saved as 'model_comparison_chart.png'")

# =====================================================================
# 5. EXPLAINABLE AI CONTRIBUTION: GRAD-CAM GENERATOR
# =====================================================================
print("\n" + "="*60)
print("[PROPOSAL CONTRIBUTION] Generating Grad-CAM Explanatory Analytics...")
print("="*60)

def make_gradcam_heatmap(img_array, target_model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [target_model.inputs], [target_model.get_layer(last_conv_layer_name).output, target_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

try:
    # Reload saved highest performing model architecture instance configuration
    print("[INFO] Reloading best saved EfficientNet checkpoint model...")
    best_saved_model = tf.keras.models.load_model("best_EfficientNetB0.keras")
    
    # Extract an index photo configuration slice
    sample_img_path = test_paths[15] 
    img = tf.keras.preprocessing.image.load_img(sample_img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Core target layer mapping target name structure for EfficientNetB0
    last_conv_layer = "top_activation"
    
    # Extract structural layers nested in base configuration
    base_functional_model = best_saved_model.layers[2]
    heatmap = make_gradcam_heatmap(img_array_exp, base_functional_model, last_conv_layer)

    # Superimpose localization metrics mapping vectors onto pixel channel sets
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Plot analytics configurations output
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original Dog Image")
    ax[0].axis("off")
    ax[1].imshow(superimposed_img)
    ax[1].set_title("Grad-CAM Spatial Focus Heatmap")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("explainable_ai_gradcam.png")
    print("[SUCCESS] Explainable AI visualization safely saved as 'explainable_ai_gradcam.png'")

except Exception as e:
    print(f"[ERROR] Could not complete Grad-CAM processing: {str(e)}")

print("\n[INFO] Script pipeline executed completely. Check directory output files for report data.")