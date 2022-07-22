#transfer learning and fine tuning of pretrained models

import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.load(name=’tf_flowers’, as_supervised=True)
NUMBER_OF_CLASSES_IN_DATASET = 5
IMG_SIZE = 160

def preprocess_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

DATASET_SIZE = 3670
BATCH_SIZE = 32
train = dataset[’train’].map(preprocess_example)
train_batches = train.shuffle(DATASET_SIZE).batch(BATCH_SIZE)

# Load MobileNetV2 model pretrained on ImageNet data
model = tf.keras.applications.MobileNetV2(
     input_shape=(IMG_SIZE, IMG_SIZE, 3),
     include_top=False, weights=’imagenet’, pooling=’avg’)
model.trainable = False

# Add a new layer for multiclass classification
new_output = tf.keras.layers.Dense(
     NUMBER_OF_CLASSES_IN_DATASET, activation=’softmax’)
new_model = tf.keras.Sequential([model, new_output])
new_model.compile(
     loss=tf.keras.losses.categorical_crossentropy,
     optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
     metrics=[’accuracy’])

# Train the classification layer
new_model.fit(train_batches.repeat(), epochs=10,
     steps_per_epoch=DATASET_SIZE // BATCH_SIZE)