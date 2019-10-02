import pickle
import time
import keras.layers
import keras.models
import keras.regularizers
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

print(train_data[3].shape)
# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std
print(pixel_mean)
print(train_data)
print(train_data[3].shape)

# One-hot encode the labels
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)
num_classes = test_labels_onehot.shape[1]

print(test_labels_onehot[3].shape)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_layers = 3
n_nodes = 500
batch_size = 32
n_epochs = 1000
dropout_rate = 0.60

n_nodes_per_layer = n_nodes // n_layers
n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))
print("Num layers: {}  total num nodes: {}".format(n_layers, n_nodes))
print("Dropout rate: {}".format(dropout_rate))

#
# Keras definitions
#

# Create a neural network model
model = keras.models.Sequential()

# First layer (need to specify the input size)
print("Adding layer with {} nodes".format(n_nodes_per_layer))
model.add(keras.layers.Dense( n_nodes_per_layer, input_shape=(n_inputs,),
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(keras.layers.Dropout(dropout_rate))

# Other hidden layers
for n in range(1, n_layers):
    print("Adding layer with {} nodes".format(n_nodes_per_layer))
    model.add(keras.layers.Dense( n_nodes_per_layer, 
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(keras.layers.Dropout(dropout_rate))

# Output layer
print("Adding layer with {} nodes".format(num_classes))
model.add(keras.layers.Dense(num_classes,
        activation='softmax',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))

# Define the optimizer
#optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.Adadelta(lr=1.0)
#optimizer = keras.optimizers.Adagrad(lr=0.01)
        
# Define cost function and optimization strategy
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

# Set callback functions to early stop training and save the best model so far
callback = [EarlyStopping(monitor='val_loss', patience=200, verbose=1),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Train the neural network
start_time = time.time()
history = model.fit(
        train_data,
        train_labels_onehot,
        epochs=n_epochs,
        callbacks = callback,
        batch_size=batch_size,
        validation_data=(test_data, test_labels_onehot),
        verbose=2,
        )

end_time = time.time()
print("Training time: ", end_time - start_time)

# Find the best costs & metrics
test_accuracy_hist = history.history['val_acc']
best_idx = test_accuracy_hist.index(max(test_accuracy_hist))
print("Max test accuracy:  {:.4f} at epoch: {}".format(test_accuracy_hist[best_idx], best_idx))

trn_accuracy_hist = history.history['acc']
best_idx = trn_accuracy_hist.index(max(trn_accuracy_hist))
print("Max train accuracy: {:.4f} at epoch: {}".format(trn_accuracy_hist[best_idx], best_idx))

test_cost_hist = history.history['val_loss']
best_idx = test_cost_hist.index(min(test_cost_hist))
print("Min test cost:  {:.5f} at epoch: {}".format(test_cost_hist[best_idx], best_idx))

trn_cost_hist = history.history['loss']
best_idx = trn_cost_hist.index(min(trn_cost_hist))
print("Min train cost: {:.5f} at epoch: {}".format(trn_cost_hist[best_idx], best_idx))

# Plot the history of the cost
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot the history of the metric
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
