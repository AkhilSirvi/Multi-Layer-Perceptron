# Multi-Layer Perceptron Neural Network

<p align="center">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Neural_Network-MLP-6366f1?style=flat-square" alt="Neural Network">
  <img src="https://img.shields.io/badge/Layers-3-10b981?style=flat-square" alt="Layers">
  <img src="https://img.shields.io/badge/Neurons-400→16→16→10-f59e0b?style=flat-square" alt="Neurons">
  <img src="https://img.shields.io/badge/Activation-TanH_|_Softmax-ec4899?style=flat-square" alt="Activation">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/AkhilSirvi/Multi-Layer-Perceptron?style=flat-square&color=22c55e" alt="License">
  <img src="https://img.shields.io/github/stars/AkhilSirvi/Multi-Layer-Perceptron?style=flat-square&color=eab308" alt="Stars">
  <img src="https://img.shields.io/github/forks/AkhilSirvi/Multi-Layer-Perceptron?style=flat-square&color=3b82f6" alt="Forks">
  <img src="https://img.shields.io/badge/PRs-Welcome-22d3ee?style=flat-square" alt="PRs Welcome">
</p>

---

A fully functional neural network system for handwritten digit recognition, built from scratch using vanilla JavaScript. This project implements a 3-layer Multi-Layer Perceptron (MLP) that can recognize digits 0-9 drawn by users.

🔗 **Live Demo:** [https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html](https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html)

<p align="center">
  <img src="https://user-images.githubusercontent.com/133025697/250265474-5206e6e2-7dcc-4a61-9d8b-cbffd7e57da6.png" alt="Neural Network System Screenshot">
</p>

---

## Table of Contents

- [Features](#features)
- [Neural Network Architecture](#neural-network-architecture)
- [How It Works](#how-it-works)
  - [Forward Propagation](#forward-propagation)
  - [Backpropagation](#backpropagation)
  - [Activation Functions](#activation-functions)
- [Data Augmentation](#data-augmentation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [License](#license)

---

## Features

- ✏️ **Interactive Drawing Canvas** - 20x20 pixel grid for drawing digits
- 🧠 **Real-time Prediction** - Instant digit recognition with confidence percentages
- 📊 **Training Visualization** - Live chart showing cost and accuracy during training
- ⚙️ **Adjustable Hyperparameters** - Modify learning rate, regularization, and training iterations
- 📱 **Responsive Design** - Works on both desktop and mobile devices
- 🔄 **Data Augmentation** - Automatic translation, rotation, and noise for better generalization
- 💾 **Pre-trained Weights** - Ready to use immediately without training

---

## Neural Network Architecture

The network uses a classic Multi-Layer Perceptron (MLP) architecture with 3 layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NETWORK ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT LAYER          HIDDEN LAYER 1      HIDDEN LAYER 2      OUTPUT       │
│   (400 neurons)        (16 neurons)        (16 neurons)        (10 neurons) │
│                                                                              │
│   ┌───┐                 ┌───┐               ┌───┐               ┌───┐       │
│   │ 1 │─────────────────│ 1 │───────────────│ 1 │───────────────│ 0 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │ 2 │─────────────────│ 2 │───────────────│ 2 │───────────────│ 1 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │ 3 │─────────────────│ 3 │───────────────│ 3 │───────────────│ 2 │       │
│   ├───┤      W1         ├───┤      W2       ├───┤      W3       ├───┤       │
│   │...│ ──────────────► │...│ ────────────► │...│ ────────────► │...│       │
│   ├───┤  (6,400 weights)├───┤ (256 weights) ├───┤ (160 weights) ├───┤       │
│   │399│                 │15 │               │15 │               │ 8 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │400│─────────────────│16 │───────────────│16 │───────────────│ 9 │       │
│   └───┘                 └───┘               └───┘               └───┘       │
│                                                                              │
│   20×20 pixel grid      TanH activation     TanH activation     Softmax     │
│   (flattened)           + Bias              + Bias              (probabilities)│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Details

| Layer | Neurons | Weights | Biases | Activation |
|-------|---------|---------|--------|------------|
| Input (A₀) | 400 | - | - | - |
| Hidden 1 (A₁) | 16 | 6,400 | 16 | TanH |
| Hidden 2 (A₂) | 16 | 256 | 16 | TanH |
| Output (A₃) | 10 | 160 | 10 | Softmax |

**Total Parameters:** 6,858 (6,816 weights + 42 biases)

---

## How It Works

### Forward Propagation

Forward propagation passes input data through the network to produce predictions:

```
For each layer l:
    Z[l] = W[l] · A[l-1] + B[l]    (weighted sum + bias)
    A[l] = g(Z[l])                  (apply activation function)
```

**Step-by-step process:**
1. **Input Layer (A₀):** Flatten the 20×20 drawing grid into a 400-element array (0 = white, 1 = black)
2. **Hidden Layer 1 (A₁):** Compute `A₁ = TanH(W₁ · A₀ + B₁)`
3. **Hidden Layer 2 (A₂):** Compute `A₂ = TanH(W₂ · A₁ + B₂)`
4. **Output Layer (A₃):** Compute `A₃ = Softmax(W₃ · A₂ + B₃)`

The output is a probability distribution over digits 0-9. The digit with the highest probability is the prediction.

### Backpropagation

Backpropagation computes gradients to update weights and biases, minimizing the cost function:

```
Cost Function: Cross-Entropy Loss
J = -(1/m) × Σ log(p_correct)

where:
- m = number of training examples
- p_correct = predicted probability of the true class
```

**Gradient Computation (backward pass):**

```
Output Layer:
    dZ₃ = A₃ - Y                    (softmax gradient with cross-entropy)
    dW₃ = (1/m) × dZ₃ᵀ · A₂
    dB₃ = (1/m) × Σ dZ₃

Hidden Layer 2:
    dZ₂ = (W₃ᵀ · dZ₃) ⊙ g'(A₂)     (⊙ = element-wise multiplication)
    dW₂ = (1/m) × dZ₂ᵀ · A₁
    dB₂ = (1/m) × Σ dZ₂

Hidden Layer 1:
    dZ₁ = (W₂ᵀ · dZ₂) ⊙ g'(A₁)
    dW₁ = (1/m) × dZ₁ᵀ · A₀
    dB₁ = (1/m) × Σ dZ₁
```

**Parameter Update (Gradient Descent with L2 Regularization):**

```
W = W - α × dW + λ × W
B = B - α × dB

where:
- α = learning rate (controls step size)
- λ = regularization coefficient (prevents overfitting)
```

### Activation Functions

#### TanH (Hyperbolic Tangent)
Used in hidden layers to introduce non-linearity:

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Output range: (-1, 1)
Derivative: g'(a) = 1 - a²
```

**Properties:**
- Zero-centered output (helps with convergence)
- Stronger gradients than sigmoid
- Saturates for very large/small inputs

#### Softmax
Used in the output layer for multi-class classification:

```
softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

Output: Probability distribution (all values sum to 1)
```

**Properties:**
- Converts raw scores to probabilities
- Each output represents P(digit = i | input)
- Highest probability = predicted digit

---

## Data Augmentation

To improve generalization and prevent overfitting, training data is augmented with random transformations:

### 1. Translation (Position Shift)
```
Shifts the image randomly by -4 to +4 pixels in both X and Y directions
Helps the model recognize digits regardless of position
```

### 2. Rotation
```
Rotates the image randomly by -15° to +15°
Helps the model recognize slightly tilted digits
Uses 2D rotation matrix:
    x' = x·cos(θ) - y·sin(θ)
    y' = x·sin(θ) + y·cos(θ)
```

### 3. Noise Addition
```
Randomly flips white pixels (0) to black (1) with 1% probability
Adds robustness against noisy input
```

---

## Project Structure

```
Multi-Layer-Perceptron/
├── src/
│   ├── index.html          # Main HTML page with UI elements
│   └── style.css           # Stylesheet for responsive design
├── scripts/
│   ├── script.js           # Main neural network logic
│   │                         - Pixel grid creation
│   │                         - Event handlers (mouse/touch)
│   │                         - Training loop (backpropagation)
│   │                         - Prediction (forward propagation)
│   │                         - Data augmentation functions
│   ├── maths.js            # Mathematical operations
│   │                         - Activation functions (TanH, Softmax)
│   │                         - Matrix operations
│   │                         - Gradient descent updates
│   ├── graph.js            # Training visualization
│   │                         - Chart.js integration
│   │                         - Cost/accuracy plotting
│   └── data.js             # Pre-trained weights & training data
│                             - Weight matrices (W1, W2, W3)
│                             - Bias vectors (B1, B2, B3)
│                             - MNIST-style training examples
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── SECURITY.md             # Security policy
```

---

## Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- No additional dependencies required!

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akhilsirvi/Multi-Layer-Perceptron.git
   ```

2. **Navigate to the project:**
   ```bash
   cd Multi-Layer-Perceptron
   ```

3. **Open in browser:**
   - Open `src/index.html` in your web browser
   - Or use a local server:
     ```bash
     # Using Python
     python -m http.server 8000
     
     # Using Node.js
     npx serve
     ```

---

## Usage

### Drawing and Recognition

1. **Draw a digit** on the 20×20 pixel grid using your mouse or touch
2. **Click "Enter your written data"** to get the prediction
3. The output panel shows:
   - The predicted digit (highest probability)
   - Confidence percentages for all digits (0-9)

### Training the Network

1. **Adjust hyperparameters** (optional):
   - Alpha (Learning Rate): Default 0.1
   - Training Length: Default 20 iterations
   - Lambda (Regularization): Default 0.000001

2. **Click "Train the neural network"**
3. Watch the progress bar and metrics update
4. The chart shows cost decreasing and accuracy increasing

---

## Hyperparameters

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| **Alpha (α)** | 0.1 | Learning rate - controls gradient descent step size | 0.001 - 1.0 |
| **Training Length** | 20 | Number of training iterations/epochs | 10 - 100 |
| **Lambda (λ)** | 0.000001 | L2 regularization coefficient | 0.000001 - 0.01 |

### Tuning Tips

- **High cost, low accuracy?** → Increase training length or learning rate
- **Overfitting (good training, poor testing)?** → Increase lambda
- **Training unstable?** → Decrease learning rate
- **Training too slow?** → Increase learning rate (carefully)

---

## Technologies Used

- **HTML5** - Structure and semantic markup
- **CSS3** - Responsive styling with Flexbox and Grid
- **JavaScript (ES6+)** - Core neural network implementation
- **Chart.js** - Training metrics visualization

### No External ML Libraries!
This project implements neural networks from scratch without TensorFlow, PyTorch, or any ML frameworks. All matrix operations, activation functions, and backpropagation are coded manually for educational purposes.

---

## Mathematical Formulas Reference

### Forward Propagation
```
A[l] = g(W[l] · A[l-1] + B[l])
```

### Cost Function (Cross-Entropy)
```
J = -(1/m) × Σᵢ Σⱼ yᵢⱼ × log(âᵢⱼ)
```

### Backpropagation Gradients
```
dW[l] = (1/m) × dZ[l] · A[l-1]ᵀ
dB[l] = (1/m) × Σ dZ[l]
dZ[l-1] = W[l]ᵀ · dZ[l] ⊙ g'(Z[l-1])
```

### Gradient Descent Update
```
W[l] := W[l] - α × dW[l] + λ × W[l]
B[l] := B[l] - α × dB[l]
```

---

## Author

**Akhil Sirvi**

- GitHub: [@akhilsirvi](https://github.com/akhilsirvi)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by the MNIST handwritten digit dataset
- Thanks to the deep learning community for educational resources
- Chart.js for the excellent charting library

---

<p align="center">
  Made with ❤️ and JavaScript
</p>
