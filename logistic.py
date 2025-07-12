# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Test the GPU
print(tf.config.list_physical_devices('GPU'))
print(tf.add(tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])).device)

# Import Keras
from tensorflow import keras

def sigmoid_centered_at_3(x):
  return 1 / (1 + np.exp(-(x - 3)))

# Example usage:
X = np.linspace(-10, 10, 1000) # Generate x values from -10 to 10
y = sigmoid_centered_at_3(X) + 0.2

# 1. Generate the test/train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("xtrain: ", X_train, len(X_train))
print("xtest: ", X_test, len(X_test))
print("ytrain: ", y_train, len(y_train))
print("ytest: ", y_test, len(y_test))

print("xtrain shape: ", X_train.shape)

# 2. Construct the LR model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile with metrics accuracy, only then will the evaluate function generate
# a test loss, test accuracy tuple, otherwise it will only return the loss
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# model = keras.Sequential([
#     keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
# ])

# # 3. Compile the model
# # For binary classification, we use 'binary_crossentropy' as the loss function
# # and 'accuracy' as a metric.
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.summary()

# 4. Visualize the model
# Plot the model
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# img = mpimg.imread('model.png')
# imgplot = plt.imshow(img)
# plt.axis('off')  # Hide axes
# plt.show()


# 5. Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.1)

# 6. Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 7. Make predictions
predictions = model.predict(X_test)

for x, y, pred in zip(X_test, y_test, predictions):
    print(f"X: {x} y: {y} Prediction: {pred}")

# 8. Convert probabilities to binary class labels (0 or 1)
# The confidence value indicates the probability of the prediction being labeled as a '1'
confidence = 0.70
predicted_classes = (predictions > confidence).astype(int)

print("\nSample predictions vs. actual labels:")
for i in range(10):
    print(f"Predicted: {predicted_classes[i]}, Actual: {y_test[i]}")
