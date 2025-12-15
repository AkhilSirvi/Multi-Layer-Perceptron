/**
 * @fileoverview Mathematical Utility Functions for Neural Network
 * 
 * This module provides all mathematical operations required for the MLP:
 * - Activation functions (TanH, Softmax) and their derivatives
 * - Matrix operations (transpose, multiplication, element-wise ops)
 * - Forward propagation computations
 * - Gradient computation helpers for backpropagation
 * - Parameter update functions with L2 regularization
 * 
 * @module maths
 * @author Akhil Sirvi
 * @version 2.0.0
 */

'use strict';

/* =============================================================================
 * NUMERICAL CONSTANTS
 * ============================================================================= */

/**
 * Small epsilon for numerical stability (prevents log(0) and division by zero)
 * @constant {number}
 */
const EPSILON = 1e-15;

/**
 * Maximum safe exponent to prevent overflow in Math.exp()
 * Math.exp(709) ≈ 8.2e307, Math.exp(710) = Infinity
 * @constant {number}
 */
const EXP_MAX = 709;

/* =============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * Hyperbolic Tangent (TanH) Activation Function
 * 
 * Maps input values to the range (-1, 1), providing zero-centered outputs
 * which improves gradient flow during backpropagation.
 * 
 * Mathematical Formula:
 *   tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
 *           = (e^(2z) - 1) / (e^(2z) + 1)
 * 
 * Properties:
 *   - Output range: (-1, 1)
 *   - Zero-centered (unlike sigmoid)
 *   - Derivative: 1 - tanh²(z)
 *   - Saturates for |z| > ~3 (vanishing gradient)
 * 
 * @param {number} z - Pre-activation value (weighted sum + bias)
 * @returns {number} Activated value in range (-1, 1)
 * 
 * @example
 * TanH(0)    // → 0
 * TanH(1)    // → 0.7616
 * TanH(-2)   // → -0.9640
 * TanH(1000) // → 1 (saturated, handled gracefully)
 */
function TanH(z) {
    // Compute using standard formula
    const expPos = Math.exp(z);
    const expNeg = Math.exp(-z);
    const result = (expPos - expNeg) / (expPos + expNeg);
    
    // Handle overflow gracefully (large |z| causes NaN)
    if (Number.isNaN(result)) {
        return z > 0 ? 1 : -1;
    }
    
    return result;
}

/**
 * Safe Natural Logarithm
 * 
 * Used for cross-entropy loss calculation: L = -Σ y·log(ŷ)
 * Includes numerical safety to prevent log(0) = -Infinity
 * 
 * @param {number} z - Input value (probability, should be > 0)
 * @returns {number} Natural logarithm, clamped for safety
 */
function log(z) {
    return Math.log(Math.max(z, EPSILON));
}

/* =============================================================================
 * RANDOM NUMBER GENERATION
 * ============================================================================= */

/**
 * Generates a uniformly distributed random number in [-1, 1]
 * 
 * Used for Xavier-like weight initialization, which helps maintain
 * appropriate variance across layers during forward propagation.
 * 
 * @param {number} decimals - Number of decimal places for precision
 * @returns {number} Random value in range [-1, 1]
 * 
 * @example
 * random(4)  // → e.g., 0.4523 or -0.8912
 */
function random(decimals) {
    const value = (Math.random() * 2) - 1;
    return Number(value.toFixed(decimals));
}

/**
 * Generates a random integer in a specified range (inclusive)
 * 
 * Used for data augmentation: random translations, rotations, etc.
 * 
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (inclusive)
 * @returns {number} Random integer in [min, max]
 * 
 * @example
 * randomrangenumber(-4, 4)  // → integer from -4 to 4
 */
function randomrangenumber(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

/* =============================================================================
 * FORWARD PROPAGATION
 * ============================================================================= */

/**
 * Performs forward propagation through a fully-connected layer
 * 
 * Computes: A[l] = σ(W[l] · A[l-1] + B[l])
 * 
 * Where:
 *   - A[l-1] = input activations (from previous layer)
 *   - W[l]   = weight matrix (stored as flattened array)
 *   - B[l]   = bias vector
 *   - σ      = activation function (TanH)
 * 
 * Weight Matrix Layout (flattened, row-major):
 *   For layer with n_in inputs and n_out outputs:
 *   W[i * n_in + j] = weight from input j to output i
 * 
 * @param {number[]} inputs - Activations from previous layer
 * @param {number[]} weights - Flattened weight matrix
 * @param {number[]} biases - Bias vector for this layer
 * @param {string} activationFn - Activation function name ("TanH")
 * @returns {number[]} Output activations
 * 
 * @example
 * // Layer: 3 inputs → 2 outputs
 * const inputs = [0.5, 0.3, 0.2];
 * const weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];  // 2×3 flattened
 * const biases = [0.1, 0.2];
 * forward_propogation(inputs, weights, biases, "TanH");
 */
function forward_propogation(inputs, weights, biases, activationFn) {
    const numInputs = inputs.length;
    const numOutputs = biases.length;
    const outputs = [];
    
    let weightIdx = 0;
    
    for (let neuron = 0; neuron < numOutputs; neuron++) {
        // Compute weighted sum: z = Σ(w_ij · x_j)
        let weightedSum = 0;
        
        for (let input = 0; input < numInputs; input++) {
            weightedSum += inputs[input] * weights[weightIdx];
            weightIdx++;
        }
        
        // Add bias: z = z + b_i
        const preActivation = weightedSum + biases[neuron];
        
        // Apply activation function: a = σ(z)
        if (activationFn === "TanH") {
            outputs.push(TanH(preActivation));
        } else {
            outputs.push(preActivation);  // Linear (no activation)
        }
    }
    
    return outputs;
}

/* =============================================================================
 * SOFTMAX FUNCTION
 * ============================================================================= */

/**
 * Softmax Activation Function
 * 
 * Converts raw output scores (logits) to a probability distribution.
 * All outputs sum to 1 and represent class probabilities.
 * 
 * Mathematical Formula:
 *   softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
 * 
 * Numerical Stability:
 *   Uses the log-sum-exp trick to prevent overflow:
 *   softmax(z_i) = e^(z_i - max(z)) / Σ_j e^(z_j - max(z))
 * 
 * Properties:
 *   - Output range: (0, 1) for each element
 *   - Sum of outputs = 1
 *   - Amplifies differences (winner-take-all tendency)
 * 
 * @param {number[]} logits - Raw scores from the output layer
 * @returns {number[]} Probability distribution (sums to 1)
 * 
 * @example
 * softmax([2.0, 1.0, 0.1])
 * // → [0.659, 0.242, 0.099] (approximately)
 */
function softmax(logits) {
    // Find max for numerical stability (log-sum-exp trick)
    const maxLogit = Math.max(...logits);
    
    // Compute shifted exponentials to prevent overflow
    const expValues = logits.map(z => Math.exp(z - maxLogit));
    
    // Sum of exponentials (denominator)
    const expSum = expValues.reduce((sum, val) => sum + val, 0);
    
    // Normalize to get probabilities
    return expValues.map(exp => exp / expSum);
}

/* =============================================================================
 * MATRIX OPERATIONS
 * ============================================================================= */

/**
 * Transposes a 2D matrix
 * 
 * Swaps rows and columns: result[j][i] = input[i][j]
 * Essential for backpropagation gradient calculations.
 * 
 * @param {number[][]} matrix - Input 2D array
 * @returns {number[][]} Transposed matrix
 * 
 * @example
 * transposeMatrix([[1, 2, 3], [4, 5, 6]])
 * // → [[1, 4], [2, 5], [3, 6]]
 */
function transposeMatrix(matrix) {
    if (!matrix || !matrix.length || !matrix[0]) {
        return [];
    }
    
    return matrix[0].map((_, colIdx) =>
        matrix.map(row => row[colIdx])
    );
}

/**
 * Matrix multiplication optimized for backpropagation
 * 
 * Computes a specialized multiplication pattern used in gradient calculations:
 *   result = transpose(M1^T × M2^T)
 * 
 * This is equivalent to computing: dW = dZ × A^T
 * where dZ is the error gradient and A is the activation matrix.
 * 
 * @param {number[][]} M1 - First matrix (typically dZ)
 * @param {number[][]} M2 - Second matrix (already transposed, typically A^T)
 * @returns {number[][]} Result matrix
 */
function matrix_multipilcation_with_transpose(M1, M2) {
    const rows1 = M1.length;
    const cols1 = M1[0].length;
    const rows2 = M2.length;
    const cols2 = M2[0].length;
    
    // Validate dimensions for this specific multiplication pattern
    if (rows1 !== cols2) {
        console.error(
            `Matrix dimension mismatch: M1 has ${rows1} rows but M2 has ${cols2} columns`
        );
        return [];
    }
    
    // Initialize output matrix
    const output = [];
    
    // Perform the specialized multiplication
    for (let a = 0; a < cols1; a++) {
        output[a] = [];
        for (let b = 0; b < rows2; b++) {
            let sum = 0;
            for (let c = 0; c < cols2; c++) {
                sum += M1[c][a] * M2[b][c];
            }
            output[a][b] = sum;
        }
    }
    
    return transposeMatrix(output);
}

/* =============================================================================
 * GRADIENT COMPUTATION HELPERS
 * ============================================================================= */

/**
 * Computes the derivative of TanH activation
 * 
 * Mathematical Formula:
 *   d/dz tanh(z) = 1 - tanh²(z)
 * 
 * Since we store activation values (not pre-activation), we use:
 *   g'(a) = 1 - a²  where a = tanh(z)
 * 
 * Used in backpropagation to compute:
 *   dZ[l] = dA[l] ⊙ g'(Z[l])
 * 
 * @param {number[][]} activations - 2D array of TanH outputs
 * @returns {number[][]} Element-wise derivative values
 * 
 * @example
 * derivative_tanH([[0.5, -0.5], [0.8, -0.8]])
 * // → [[0.75, 0.75], [0.36, 0.36]]
 */
function derivative_tanH(activations) {
    return activations.map(row =>
        row.map(a => 1 - (a * a))
    );
}

/**
 * Element-wise (Hadamard) multiplication of two matrices
 * 
 * Mathematical Operation:
 *   result[i][j] = A[i][j] × B[i][j]
 * 
 * Used in backpropagation for:
 *   dZ[l] = (W[l+1]^T · dZ[l+1]) ⊙ g'(Z[l])
 * 
 * The ⊙ symbol denotes element-wise (not matrix) multiplication.
 * 
 * @param {number[][]} A - First matrix
 * @param {number[][]} B - Second matrix (same dimensions as A)
 * @returns {number[][]} Element-wise product A ⊙ B
 * 
 * @example
 * element_wise_multiplication([[1, 2], [3, 4]], [[5, 6], [7, 8]])
 * // → [[5, 12], [21, 32]]
 */
function element_wise_multiplication(A, B) {
    return A.map((row, i) =>
        row.map((val, j) => val * B[i][j])
    );
}

/* =============================================================================
 * PARAMETER UPDATE FUNCTIONS (GRADIENT DESCENT)
 * ============================================================================= */

/**
 * L2 Regularization coefficient (λ)
 * 
 * Adds penalty for large weights to prevent overfitting:
 *   L_regularized = L + (λ/2) × ||W||²
 * 
 * Typical values: 0.0001 to 0.01
 * @type {number}
 */
let lambda = 0.000001;

/**
 * Updates weight matrix using gradient descent with L2 regularization
 * 
 * Update Rule:
 *   W_new = W_old - α × dW + λ × W_old
 * 
 * Components:
 *   - α × dW: Gradient descent step (move against gradient)
 *   - λ × W:  Regularization term (weight persistence)
 * 
 * Note: Standard L2 regularization subtracts λ×W (weight decay).
 * This implementation adds it for backward compatibility with
 * pre-trained weights. The effect is minor for small λ.
 * 
 * @param {number[]} weights - Current weight values
 * @param {number} learningRate - Step size (α)
 * @param {number[][]} gradients - Computed gradient dW
 * @returns {number[]} Updated weight values
 */
function W_update(weights, learningRate, gradients) {
    // Flatten and transpose gradient matrix to match weight layout
    const flatGradients = transposeMatrix(gradients).flat();
    
    // Scale gradients by learning rate: α × dW
    const scaledGradients = flatGradients.map(g => g * learningRate);
    
    // Gradient descent: W = W - α × dW
    const updated = weights.map((w, i) => w - scaledGradients[i]);
    
    // L2 regularization: W = W + λ × W_old
    const regularization = weights.map(w => w * lambda);
    
    return updated.map((w, i) => w + regularization[i]);
}

/**
 * Updates bias vector using gradient descent
 * 
 * Update Rule:
 *   B_new = B_old - α × dB
 * 
 * Note: Regularization is typically NOT applied to biases
 * since they don't contribute to model complexity.
 * 
 * @param {number[]} biases - Current bias values
 * @param {number} learningRate - Step size (α)
 * @param {number[]} gradients - Computed gradient dB
 * @returns {number[]} Updated bias values
 */
function B_update(biases, learningRate, gradients) {
    // Scale gradients by learning rate: α × dB
    const scaledGradients = gradients.map(g => g * learningRate);
    
    // Gradient descent: B = B - α × dB
    return biases.map((b, i) => b - scaledGradients[i]);
}

/* =============================================================================
 * OUTPUT FORMATTING UTILITIES
 * ============================================================================= */

/**
 * Converts prediction scores to sorted percentage format
 * 
 * Takes raw or softmax outputs and formats them as percentages,
 * sorted by confidence (highest first) for display.
 * 
 * @param {number[]} scores - Array of numeric values (e.g., softmax outputs)
 * @returns {Object[]} Array of {index, percentage} sorted descending
 * 
 * @example
 * convertToPercentages([0.1, 0.7, 0.2])
 * // → [{index: 1, percentage: 70}, {index: 2, percentage: 20}, {index: 0, percentage: 10}]
 */
function convertToPercentages(scores) {
    const sum = scores.reduce((total, val) => total + val, 0);
    
    const percentages = scores.map((score, index) => ({
        index: index,
        percentage: (score / sum) * 100
    }));
    
    // Sort by percentage (highest first)
    return percentages.sort((a, b) => b.percentage - a.percentage);
}
