import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\n🎉 Dataset Loaded Successfully!\n")

print("Number of Classes:", len(dataset.class_names))

print("\nClasses Found:")

for i, name in enumerate(dataset.class_names):
    print(f"{i+1}. {name}")