# Internship Task - 2
# Deep Learning Project
# ========================================================

# Step 1: Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 2: Load the MNIST dataset
# TensorFlow provides this dataset directly
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 3: Normalize the image data
# Scaling pixel values to the range [0, 1] for faster convergence
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 4: Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),     
    tf.keras.layers.Dense(128, activation='relu'),     
    tf.keras.layers.Dense(10, activation='softmax')    
])

# Step 5: Compile the model
# Define optimizer, loss function, and performance metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
# Model learns patterns from training data
print(" Training the model...")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)

# Step 7: Evaluate the model on test data
# Check how well the model performs on unseen data
print("\n Evaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f" Test Accuracy: {test_acc * 100:.2f}%")

# Step 8: Make predictions
# Model predicts the labels for test images
predictions = model.predict(x_test)

# Step 9: Visualize one image and its prediction
plt.figure(figsize=(4,4))
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted Label: {tf.argmax(predictions[0]).numpy()}")
plt.axis('off')
plt.show()

# Optional: Display training accuracy graph
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
