/**
 * Mathematical Utility Functions for Neural Network
 * 
 * This module contains all mathematical operations required for:
 * - Activation functions (TanH, Softmax)
 * - Matrix operations (transpose, multiplication)
 * - Forward propagation
 * - Gradient computation helpers
 * - Parameter update functions
 * 
 * @module maths
 */

/* =============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * Hyperbolic Tangent (TanH) Activation Function
 * Maps input values to range (-1, 1)
 * Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
 * 
 * Handles edge cases where large values cause NaN results
 * 
 * @param {number} z - Input value (pre-activation)
 * @returns {number} Activated value in range (-1, 1)
 */
function TanH(z) {
  let tanhvalue = (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
  
  // Handle overflow cases for very large positive/negative values
  if (isNaN(tanhvalue) && z > 0) {
    return 1;   // Large positive values saturate to 1
  } else if (isNaN(tanhvalue) && z < 0) {
    return -1;  // Large negative values saturate to -1
  } else {
    return tanhvalue;
  }
}

/**
 * Natural Logarithm wrapper
 * Used for cross-entropy loss calculation: -log(p)
 * 
 * @param {number} z - Input value (should be > 0)
 * @returns {number} Natural log of z
 */
function log(z) {
  return Math.log(z);
}

/* =============================================================================
 * RANDOM NUMBER GENERATION
 * ============================================================================= */

/**
 * Generates a random number between -1 and 1
 * Used for weight initialization (Xavier-like initialization)
 * 
 * @param {number} decimals - Number of decimal places to round to
 * @returns {number} Random value in range [-1, 1]
 */
function random(decimals) {
  var randomNumber = Math.random() * 2 - 1;  // Scale [0,1] to [-1,1]
  return parseFloat(randomNumber.toFixed(decimals));
}

/* =============================================================================
 * FORWARD PROPAGATION
 * ============================================================================= */

/**
 * Performs forward propagation for a single layer
 * Computes: A[l] = activation(W[l] · A[l-1] + B[l])
 * 
 * @param {Array} M1 - Input activations from previous layer (1D array)
 * @param {Array} M2 - Weight matrix (flattened 1D array)
 * @param {Array} B - Bias vector for this layer
 * @param {string} Activation_function - Activation type ("TanH")
 * @returns {Array} Output activations for this layer
 */
function forward_propogation(M1, M2, B, Activation_function) {
  let multiplication_data = 0;
  let data_array = [];
  let shape_first_matrix_row = M1.length;  // Number of inputs
  let k = 0;  // Index into input array
  let v = 0;  // Index into bias array / output neuron counter
  
  // Iterate through all weights
  for (let i = 0; i < M2.length; i++) {
    // Accumulate weighted sum: sum += input[k] * weight[i]
    multiplication_data = multiplication_data + M1[k] * M2[i];
    k++;
    
    // When we've processed all inputs for one output neuron
    if (k == shape_first_matrix_row) {
      k = 0;  // Reset input index for next output neuron
      
      // Apply activation function with bias and store result
      if (Activation_function == "TanH") {
        data_array.push(TanH(multiplication_data + B[v]));
        v++;
      }
      multiplication_data = 0;  // Reset accumulator
    }
  }
  return data_array;
}

/* =============================================================================
 * SOFTMAX FUNCTION
 * ============================================================================= */

/**
 * Softmax Activation Function
 * Converts raw output scores to probability distribution
 * Formula: softmax(z_i) = e^(z_i) / Σ e^(z_j)
 * 
 * Used in the output layer for multi-class classification
 * Output values sum to 1 and can be interpreted as probabilities
 * 
 * @param {Array} z - Raw output scores from last layer
 * @returns {Array} Probability distribution (values sum to 1)
 */
function softmax(z) {
  let data = [];
  let expsum = 0;
  
  // Calculate sum of exponentials (denominator)
  for (let k = 0; k < z.length; k++) {
    expsum += Math.exp(z[k]);
  }
  
  // Calculate softmax for each element
  for (let i = 0; i < z.length; i++) {
    data.push(Math.exp(z[i]) / expsum);
  }
  return data;
}

/* =============================================================================
 * MATRIX OPERATIONS
 * ============================================================================= */

/**
 * Transposes a 2D matrix (swaps rows and columns)
 * Used for gradient computations in backpropagation
 * 
 * @param {Array} matrix - 2D array to transpose
 * @returns {Array} Transposed matrix where result[j][i] = input[i][j]
 */
function transposeMatrix(matrix) {
  const transposedMatrix = matrix[0].map((_, colIndex) =>
    matrix.map((row) => row[colIndex])
  );
  return transposedMatrix;
}

/**
 * Matrix multiplication with transpose handling
 * Computes: result = M1^T × M2^T (then transposes result)
 * 
 * Used for computing gradients: dW = dZ × A^T
 * 
 * @param {Array} M1 - First matrix (2D array)
 * @param {Array} M2 - Second matrix (2D array, already transposed)
 * @returns {Array} Result of matrix multiplication
 */
function matrix_multipilcation_with_transpose(M1, M2) {
  let output = [];
  let rows1 = M1.length;
  let rows2 = M2.length;
  let colum1 = M1[0].length;
  let colum2 = M2[0].length;
  
  // Validate matrix dimensions for multiplication
  if (rows1 !== colum2) {
    console.log(
      "Error: The number of columns in Matrix 1 must match the number of rows in Matrix 2."
    );
    return output;
  }
  
  // Perform matrix multiplication
  for (let a = 0; a < colum1; a++) {
    output[a] = [];
    for (let b = 0; b < rows2; b++) {
      let sum = 0;
      for (let c = 0; c < colum2; c++) {
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
 * Computes derivative of TanH activation
 * Derivative: d/dz tanh(z) = 1 - tanh²(z)
 * 
 * Since we have the activation values (not pre-activation), we compute:
 * g'(a) = 1 - a² where a = tanh(z)
 * 
 * @param {Array} h - 2D array of TanH activation values
 * @returns {Array} Element-wise derivative values
 */
function derivative_tanH(h) {
  // Square each activation value: a²
  let square_matrix = h.map((innerArr) => innerArr.map((value) => value ** 2));
  
  // Compute 1 - a² for each element
  let newArray = square_matrix.map((subArray) =>
    subArray.map((value) => 1 - value)
  );
  return newArray;
}

/**
 * Element-wise (Hadamard) multiplication of two matrices
 * Used in backpropagation: dZ = (W^T · dZ_next) ⊙ g'(Z)
 * 
 * @param {Array} J1 - First matrix (2D array)
 * @param {Array} J2 - Second matrix (2D array, same dimensions as J1)
 * @returns {Array} Element-wise product J1 ⊙ J2
 */
function element_wise_multiplication(J1, J2) {
  const result = J1.map((row, i) => row.map((num, j) => num * J2[i][j]));
  return result;
}

/* =============================================================================
 * PARAMETER UPDATE FUNCTIONS (GRADIENT DESCENT)
 * ============================================================================= */

/** 
 * L2 Regularization coefficient (lambda)
 * Helps prevent overfitting by penalizing large weights
 * Update becomes: W = W - α*(dW + λ*W)
 */
let lambda = 0.000001;

/**
 * Updates weight matrix using gradient descent with L2 regularization
 * Formula: W_new = W_old - α*dW + λ*W_old
 * 
 * The regularization term (λ*W) encourages smaller weights,
 * which helps prevent overfitting
 * 
 * @param {Array} intialmatrix - Current weight values (1D array)
 * @param {number} learning_rate - Step size (alpha)
 * @param {Array} backpropvalue - Computed gradient dW (2D array)
 * @returns {Array} Updated weight values
 */
function W_update(intialmatrix, learning_rate, backpropvalue) {
  // Flatten and transpose the gradient matrix
  let flat_backpropvalue = transposeMatrix(backpropvalue).flat();
  
  // Scale gradients by learning rate: α * dW
  let multipliedMatrix = flat_backpropvalue.map(function (value) {
    return value * learning_rate;
  });
  
  // Gradient descent step: W = W - α*dW
  let result = intialmatrix.map(
    (value, index) => value - multipliedMatrix[index]
  );

  // L2 regularization term: λ * W
  let regulazationmatrix = intialmatrix.map(function (value) {
    return value * lambda;
  });

  // Add regularization: W = W - α*dW + λ*W
  let regulazationanswer = result.map((value, index) => value + regulazationmatrix[index]);

  return regulazationanswer;
}

/**
 * Updates bias vector using gradient descent
 * Formula: B_new = B_old - α*dB
 * 
 * Note: Regularization is typically not applied to biases
 * 
 * @param {Array} intialmatrix - Current bias values (1D array)
 * @param {number} learning_rate - Step size (alpha)
 * @param {Array} backpropvalue - Computed gradient dB (1D array)
 * @returns {Array} Updated bias values
 */
function B_update(intialmatrix, learning_rate, backpropvalue) {
  // Scale gradients by learning rate: α * dB
  let multipliedMatrix = backpropvalue.map(function (value) {
    return value * learning_rate;
  });
  
  // Gradient descent step: B = B - α*dB
  let result = intialmatrix.map(
    (value, index) => value - multipliedMatrix[index]
  );
  return result;
}

/* =============================================================================
 * OUTPUT FORMATTING UTILITIES
 * ============================================================================= */

/**
 * Converts an array of values to percentage format
 * Sorts results by percentage in descending order
 * Used to display prediction confidence for each digit
 * 
 * @param {Array} arr - Array of numeric values (e.g., softmax outputs)
 * @returns {Array} Array of objects with index and percentage, sorted descending
 */
function convertToPercentages(arr) {
  // Calculate total sum
  const sum = arr.reduce((total, num) => total + num, 0);
  
  // Convert each value to percentage with its original index
  const percentages = arr.map((num, index) => ({
    index: index,
    percentage: (num / sum) * 100,
  }));
  
  // Sort by percentage (highest first)
  percentages.sort((a, b) => b.percentage - a.percentage);
  return percentages;
}
