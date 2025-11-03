## Deep Learning Project
*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : JAGAN PADHIARY

*INTERN ID* : CT04DR268

*DOMAIN* : DATA SCIENCE

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH KUMAR
## PROJECT OVERVIEW
This project demonstrates a **deep learning model** built using **TensorFlow** to classify handwritten digits from the **MNIST dataset**.  
It is a part of an **internship task** focusing on understanding the end-to-end workflow of building, training, and evaluating a neural network using image data.

---

##  Features
- Loads and preprocesses the MNIST dataset automatically  
- Builds a **fully connected neural network (ANN)**  
- Trains the model on 60,000 handwritten digit images  
- Evaluates accuracy on 10,000 test images  
- Visualizes predictions with Matplotlib  

---

##  Model Architecture
| Layer Type | Description |
| Flatten | Converts 28×28 pixel images into a 1D vector |
| Dense (128 units) | Fully connected layer with ReLU activation |
| Dense (10 units) | Output layer using softmax activation for 10 classes (digits 0–9) |

---

##  Tech Stack
- **Python 3.13.5**
- **TensorFlow / Keras**
- **Matplotlib**

---
MNIST_DeepLearning_Project
│
├── mnist_project.py # Main deep learning script
├── README.md # Project documentation
└── (auto-downloaded MNIST dataset)


---

##  How It Works

### Step 1: Load Data
The MNIST dataset (70,000 grayscale images of digits 0–9) is loaded directly from TensorFlow.

### Step 2: Preprocess Data
Each image is normalized (pixel values divided by 255.0) to help the model train efficiently.

### Step 3: Build Model
A simple sequential model with dense layers is defined using TensorFlow’s Keras API.

### Step 4: Train Model
The model is trained for 5 epochs using the Adam optimizer and sparse categorical crossentropy loss.

### Step 5: Evaluate and Predict
The model’s accuracy is evaluated, and predictions are visualized for random test images.

---
This will:

Train the model on MNIST data

Display the test accuracy

Show a visualization of a test image and its predicted label

## Example Output

Training Output:

Epoch 1/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.26 - accuracy: 0.93
...
Test Accuracy: 97.9%


Visualization:
An MNIST digit image is displayed with its predicted label on top.

## Future Improvements

Add Convolutional Neural Networks (CNNs) for higher accuracy

Implement Dropout layers to reduce overfitting

Save and load trained models using model.save() and tf.keras.models.load_model()

Add a Flask or Streamlit web interface for digit prediction
##  Project Structure

<img width="292" height="306" alt="Image" src="https://github.com/user-attachments/assets/8c4eca44-a466-4dee-bc1a-af0eff98d45a" />


<img width="439" height="316" alt="Image" src="https://github.com/user-attachments/assets/5d0f612b-bcba-4353-9484-4c4ca05d41e1" />
