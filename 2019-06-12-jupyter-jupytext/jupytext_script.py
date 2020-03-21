# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from itertools import cycle

# %%
import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from tensorflow.python.keras.utils.vis_utils import model_to_dot

# %%
print(tf.VERSION)
tf.keras.backend.clear_session()

# %%
# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# %%
num_train_examples = len(y_train)
num_test_examples = len(y_test)

# %% [markdown]
# # Config
# Configuration required for training

# %%
# training configuration
MAX_LR = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_TRAIN_STEPS = int(num_train_examples / BATCH_SIZE)
NUM_TEST_STEPS = int(num_test_examples / BATCH_SIZE)
FILTERS = [16, 32, 64, 64]
OPTIMIZER = tf.keras.optimizers.SGD(lr=MAX_LR, momentum=0.9, nesterov=True)
LOSS = tf.keras.losses.categorical_crossentropy
METRICS = [
    tf.keras.metrics.categorical_accuracy,
]


# %%
# Load functions
def preprocess_example(img, label):
    img = tf.expand_dims(img, axis=-1)
    img = tf.cast(img, tf.float32)
    img /= 255.

    label = tf.one_hot(label, depth=10)

    return img, label


# %%
def train_dataset_fn():
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).repeat() \
        .map(preprocess_example) \
        .batch(64) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


# %%
def test_dataset_fn():
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.repeat().map(preprocess_example).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    return test_dataset


# %%
# get keras model
def mini_cnn(input_shape,
             num_classes,
             filters=[16, 32, 64, 64],
             activation="softmax",
             model_name="basic_cnn4",
             **conv_kwargs):
    use_bias = False

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters[0], (3, 3), padding="same", use_bias=use_bias, **conv_kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters[1], (3, 3), padding="same", use_bias=use_bias, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters[2], (3, 3), padding="same", use_bias=use_bias, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(filters[3], use_bias=use_bias, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    logits = tf.keras.layers.Dense(
        num_classes, use_bias=True, name="logits", bias_initializer=tf.keras.initializers.constant(0.1))(x)
    probas = tf.keras.layers.Activation(activation=activation, name="probas")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=probas, name=model_name)

    return model


# %%
def learning_rate_scheduler(epoch, warmup=10):
    _epoch = epoch + 1
    if _epoch < 10:
        return MAX_LR * float(_epoch / warmup)
    else:
        c = (_epoch - warmup) / (NUM_EPOCHS - warmup)

        return MAX_LR * (1. / 2.) * (1. + np.cos(np.pi * c))


# %%
# get model
keras_model = mini_cnn(
    input_shape=(28, 28, 1),
    num_classes=10,
    filters=FILTERS,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    model_name="fashion_mnist_test")

# %%
CALLBACKS = []

# %%
# get visualisation
SVG(model_to_dot(keras_model, show_shapes=True).create(prog='dot', format='svg'))

# %%
# compile
keras_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

# %%
# fit
history = keras_model.fit(
    train_dataset_fn(),
    epochs=NUM_EPOCHS,
    verbose=2,
    callbacks=CALLBACKS,
    validation_data=test_dataset_fn(),
    initial_epoch=0,
    steps_per_epoch=NUM_TRAIN_STEPS,
    validation_steps=NUM_TEST_STEPS)

# %%
# Plot training & validation accuracy values
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%
# Get predictions
x_test_ = np.expand_dims(x_test, axis=-1) / 255.
y_true = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_pred = keras_model.predict(x_test_, batch_size=64)

# %%
# Compute pr curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

# %%
# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
average_precision["micro"] = average_precision_score(y_true, y_pred, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

# %%
# Plot pr curve
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

# %%
plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

# %%
lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})' ''.format(average_precision["micro"]))

# %%
for i, color in zip(range(10), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})' ''.format(i, average_precision[i]))

# %%
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -1.0), prop=dict(size=14))
plt.show()
