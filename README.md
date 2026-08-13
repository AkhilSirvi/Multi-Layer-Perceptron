# Multi-Layer Perceptron Neural Network


A fully functional neural network system for handwritten digit recognition, built from scratch using vanilla JavaScript. This project implements a 3-layer Multi-Layer Perceptron (MLP) that can recognize digits 0-9 drawn by users.

🔗 **Live Demo:** [https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html](https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html)

<p align="center">
  <img width="1119" height="578" alt="image" src="https://github.com/user-attachments/assets/abd61d8f-80aa-41c6-b1d5-8b85ad8890ea" />
</p>


## Neural Network Architecture

The network uses a classic Multi-Layer Perceptron (MLP) architecture with 3 layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NETWORK ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT LAYER          HIDDEN LAYER 1      HIDDEN LAYER 2      OUTPUT       │
│   (784 neurons)        (128 neurons)       (64 neurons)        (10 neurons) │
│                                                                             │
│   ┌───┐                 ┌───┐               ┌───┐               ┌───┐       │
│   │ 1 │─────────────────│ 1 │───────────────│ 1 │───────────────│ 0 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │ 2 │─────────────────│ 2 │───────────────│ 2 │───────────────│ 1 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │ 3 │─────────────────│ 3 │───────────────│ 3 │───────────────│ 2 │       │
│   ├───┤      W1         ├───┤      W2       ├───┤      W3       ├───┤       │
│   │...│ ──────────────► │...│ ────────────► │...│ ────────────► │...│       │
│   ├───┤(100,352 weights)├───┤(8192 weights) ├───┤(640 weights)  ├───┤       │
│   │783│                 │127│               │63 │               │ 8 │       │
│   ├───┤                 ├───┤               ├───┤               ├───┤       │
│   │784│─────────────────│128│───────────────│64 │───────────────│ 9 │       │
│   └───┘                 └───┘               └───┘               └───┘       │
│                                                                             │
│   28×28 pixel grid      Leaky ReLU         Leaky ReLU           Softmax     │
│   (flattened)           + Bias              + Bias           (probabilities)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Details

| Layer         | Neurons | Weights | Biases | Activation |
|---------------|---------|---------|--------|------------|
| Input (A₀)    | 784     | -       | -      | -          |
| Hidden 1 (A₁) | 128     | 100,352 | 128    | Leaky ReLU |
| Hidden 2 (A₂) | 64      | 8192    | 64     | Leaky ReLU |
| Output (A₃)   | 10      | 640     | 10     | Softmax    |

**Total Parameters:** 109,386 (109,184 weights + 202 biases)

---

## How It Works

### Forward Propagation

Forward propagation passes input data through the network to produce predictions:

### Layer Computation

For each layer $l$:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = g\left(Z^{[l]}\right)
$$

**Step-by-step process:**
1. **Input Layer (A₀):** Flatten the 28×28 drawing grid into a 784-element array (0 = white, 1 = black)
2. **Hidden Layer 1 (A₁):** Compute `A₁ = TanH(W₁ · A₀ + B₁)`
3. **Hidden Layer 2 (A₂):** Compute `A₂ = TanH(W₂ · A₁ + B₂)`
4. **Output Layer (A₃):** Compute `A₃ = Softmax(W₃ · A₂ + B₃)`

The output is a probability distribution over digits 0-9. The digit with the highest probability is the prediction.

### Backpropagation

Backpropagation computes gradients to update weights and biases, minimizing the cost function:

### Cost Function — Cross-Entropy Loss

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \log\left(p_{\text{correct}}^{(i)}\right)
$$

where  

- $m$ = number of training examples  
- $p_{\text{correct}}^{(i)}$ = predicted probability of the true class for example $i$

**Gradient Computation (backward pass):**

### Backpropagation

#### Output Layer

$$
dZ^{[3]} = A^{[3]} - Y
$$

$$
dW^{[3]} = \frac{1}{m} \, dZ^{[3]} (A^{[2]})^T
$$

$$
db^{[3]} = \frac{1}{m} \sum dZ^{[3]}
$$

---

#### Hidden Layer 2

$$
dZ^{[2]} = (W^{[3]})^T dZ^{[3]} \odot g'(A^{[2]})
$$

$$
dW^{[2]} = \frac{1}{m} \, dZ^{[2]} (A^{[1]})^T
$$

$$
db^{[2]} = \frac{1}{m} \sum dZ^{[2]}
$$

---

#### Hidden Layer 1

$$
dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot g'(A^{[1]})
$$

$$
dW^{[1]} = \frac{1}{m} \, dZ^{[1]} (A^{[0]})^T
$$

$$
db^{[1]} = \frac{1}{m} \sum dZ^{[1]}
$$

**Parameter Update (Gradient Descent with L2 Regularization):**

### Parameter Update Rule

$$
W = W - \alpha\, dW - \lambda\, W
$$

$$
B = B - \alpha\, dB
$$

where  

- $\alpha$ = learning rate (controls step size)  
- $\lambda$ = regularization coefficient (prevents overfitting)

### Activation Functions

#### TanH (Hyperbolic Tangent)
Used in hidden layers to introduce non-linearity:

### Tanh Activation

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

Output range: $(-1, 1)$

Derivative:

$$
g'(z) = 1 - \tanh^2(z)
$$


#### Softmax
Used in the output layer for multi-class classification:

### Softmax Function

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Output: Probability distribution (all values sum to 1)

## Data Augmentation

To improve generalization and prevent overfitting, training data is augmented with random transformations.

---

### No External ML Libraries
This project implements neural networks from scratch without any ML frameworks.

## Mathematical Formulas Reference

### Forward Propagation

$$
A^{[l]} = g\left(W^{[l]} A^{[l-1]} + b^{[l]}\right)
$$

### Cost Function (Cross-Entropy)

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{n} y_{ij} \log(\hat{a}_{ij})
$$

### Backpropagation Gradients

$$
dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T
$$

$$
db^{[l]} = \frac{1}{m} \sum dZ^{[l]}
$$

$$
dZ^{[l-1]} = (W^{[l]})^T dZ^{[l]} \odot g'(Z^{[l-1]})
$$

### Gradient Descent Update

$$
W^{[l]} := W^{[l]} - \alpha dW^{[l]} - \lambda W^{[l]}
$$

$$
b^{[l]} := b^{[l]} - \alpha db^{[l]}
$$

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
