# Multi-Layer Perceptron Neural Network

MLP network for handwritten digit recognition, built from scratch using and no ml libraries.

**Live Demo:** [https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html](https://akhilsirvi.github.io/Multi-Layer-Perceptron/src/index.html)

<p align="center">
  <img width="1119" height="578" alt="image" src="https://github.com/user-attachments/assets/abd61d8f-80aa-41c6-b1d5-8b85ad8890ea" />
</p>


## Neural Network

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
│   28×28 pixel grid      activation func     activation func     Softmax     │
│                         + Bias              + Bias                          │
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

### Forward Propagation

For each layer $l$:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = g\left(Z^{[l]}\right)
$$
### Backpropagation

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

- $\alpha$ = learning rate
- $\lambda$ = regularization coefficient

## Data Augmentation

To add noise to data for more learning

---


## Author

**Akhil Sirvi**

- GitHub: [@akhilsirvi](https://github.com/akhilsirvi)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
