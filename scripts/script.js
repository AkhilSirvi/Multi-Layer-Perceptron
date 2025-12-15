/**
 * @fileoverview Multi-Layer Perceptron (MLP) Neural Network for Digit Recognition
 * 
 * A 3-layer neural network that recognizes handwritten digits (0-9) drawn by users.
 * Implements complete forward propagation, backpropagation, and gradient descent.
 * 
 * Network Architecture:
 *   Input Layer (A_0):  400 neurons (20×20 pixel grid, flattened)
 *   Hidden Layer 1 (A_1): 16 neurons with TanH activation
 *   Hidden Layer 2 (A_2): 16 neurons with TanH activation
 *   Output Layer (A_3):  10 neurons (digits 0-9) with Softmax
 * 
 * Training Algorithm:
 *   - Loss Function: Cross-Entropy
 *   - Optimization: Mini-batch Gradient Descent with L2 Regularization
 *   - Data Augmentation: Translation, Rotation, Noise
 * 
 * @author Akhil Sirvi
 * @version 2.0.0
 */

'use strict';

/* =============================================================================
 * NEURAL NETWORK ARCHITECTURE CONFIGURATION
 * ============================================================================= */

/**
 * Network Architecture Constants
 * Defines the structure of the Multi-Layer Perceptron
 */
const NETWORK_CONFIG = Object.freeze({
    INPUT_SIZE: 400,    // 20×20 pixel grid (flattened)
    HIDDEN_1_SIZE: 16,  // First hidden layer neurons
    HIDDEN_2_SIZE: 16,  // Second hidden layer neurons
    OUTPUT_SIZE: 10,    // Digit classes (0-9)
    GRID_WIDTH: 20,     // Drawing grid width
    GRID_HEIGHT: 20     // Drawing grid height
});

/* =============================================================================
 * DOM ELEMENT REFERENCES
 * ============================================================================= */

/** Container for the drawing grid where users draw digits */
let second_box = document.getElementById("second");

/** Display area for showing the neural network's prediction output */
let output_text = document.getElementById("output_text");

/** Progress bar element showing training completion */
let loadbar = document.getElementById("loadbarmain");

/** Container for the loading/progress bar */
let loadbarcontain = document.getElementById("loadbar");

/** Display area for showing cost and accuracy values during training */
let cost_value_box = document.getElementById("cost_value_output");

/** Input field for the learning rate (alpha) parameter */
let alpha_value = document.getElementById("alpha_value");

/** Input field for specifying the number of training iterations */
let train_length_input = document.getElementById("train_length_input");

/** Submit button for applying hyperparameter changes */
let alpha_submit = document.getElementById("alpha_submit");

/* =============================================================================
 * PIXEL GRID CREATION
 * ============================================================================= */

/**
 * Creates the interactive pixel grid for drawing digits
 * 
 * Generates a 20×20 grid (400 pixels) where users can draw.
 * Each pixel is a div element that toggles between white (0) and black (1).
 * This grid forms the input layer of the neural network.
 */
function pixel_creater() {
    const totalPixels = NETWORK_CONFIG.GRID_WIDTH * NETWORK_CONFIG.GRID_HEIGHT;
    
    for (let i = 0; i < totalPixels; i++) {
        const pixel = document.createElement("div");
        pixel.className = "box";
        pixel.draggable = false;
        pixel.dataset.index = i;  // Store pixel index for debugging
        second_box.appendChild(pixel);
    }
}

// Initialize the drawing grid on page load
pixel_creater();

/* =============================================================================
 * MOUSE/TOUCH EVENT HANDLING FOR DRAWING
 * ============================================================================= */

/** Reference to the document body for global event listeners */
let body = document.body;

/** NodeList of all pixel elements in the drawing grid */
let button = document.querySelectorAll(".box");

/** Flag to track if the left mouse button is currently pressed */
let click = false;

/**
 * Mouse down event listener
 * Sets click flag to true when left mouse button (button 1) is pressed
 */
body.addEventListener("mousedown", (event) => {
  if (event.buttons === 1) {
    click = true;
  }
});

/**
 * Mouse up event listener
 * Resets click flag when left mouse button (button 0) is released
 */
body.addEventListener("mouseup", (event) => {
  if (event.button === 0) {
    click = false;
  }
});

/**
 * Attach drawing event listeners to each pixel in the grid
 * Supports both mouse (desktop) and touch (mobile) interactions
 */
button.forEach((button) => {
  // Desktop: Draw on single click (for individual pixels)
  button.addEventListener("mousedown", (event) => {
    if (event.buttons === 1) {
      button.style.background = "black";
    }
  });

  // Desktop: Draw when hovering with mouse button pressed (for dragging)
  button.addEventListener("mouseover", () => {
    if (click == true) {
      button.style.background = "black";
    }
  });

  // Mobile: Handle touch-based drawing
  second_box.addEventListener("touchmove", (event) => {
    event.preventDefault(); // Prevent scrolling while drawing
    const touch = event.touches[0];
    const x = touch.clientX;
    const y = touch.clientY;
    const rect = button.getBoundingClientRect();

    // Check if touch position is within this pixel's bounds
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      button.style.background = "black";
    }
  });
});

/* =============================================================================
 * LAYER DIMENSIONS & ACTIVATION ARRAYS
 * ============================================================================= */

/**
 * Layer dimensions (used throughout the network)
 * Kept as variables for backward compatibility with existing code
 */
let A_0_length = NETWORK_CONFIG.INPUT_SIZE;
let A_1_length = NETWORK_CONFIG.HIDDEN_1_SIZE;
let A_2_length = NETWORK_CONFIG.HIDDEN_2_SIZE;
let A_3_length = NETWORK_CONFIG.OUTPUT_SIZE;

/** 
 * Activation arrays for each layer
 * Populated during forward propagation, reset between predictions
 */
let A_0 = [];  // Input layer activations (pixel values: 0 or 1)
let A_1 = [];  // Hidden layer 1 activations (after TanH)
let A_2 = [];  // Hidden layer 2 activations (after TanH)
let A_3 = [];  // Output layer activations (before softmax)

/**
 * Learning Rate (α/alpha)
 * 
 * Controls the step size during gradient descent optimization.
 * - Smaller values (0.001-0.01): Slower but more stable convergence
 * - Larger values (0.1-1.0): Faster but potentially unstable/overshooting
 * 
 * Recommended: Start with 0.01-0.1, adjust based on loss curve
 * @type {number}
 */
let alpha = 0.1;

/* =============================================================================
 * WEIGHT AND BIAS INITIALIZATION FUNCTIONS
 * ============================================================================= */

/**
 * Initialize weights for Layer 1 (Input -> Hidden1)
 * Creates W_1_length random values between -1 and 1
 * Shape: A_1_length x A_0_length = 16 x 400 = 6,400 weights
 */
function W_1_function_random_no() {
  let W_1_length = A_1_length * A_0_length;
  for (let i = 0; i < W_1_length; i++) {
    W_1.push(random(2));
  }
}

/**
 * Initialize biases for Layer 1
 * Creates one bias value per neuron in the first hidden layer
 */
function B_1_function_random_no() {
  let B_1_length = A_1_length;
  for (let i = 0; i < B_1_length; i++) {
    B_1.push(random(2));
  }
}

/**
 * Initialize weights for Layer 2 (Hidden1 -> Hidden2)
 * Shape: A_2_length x A_1_length = 16 x 16 = 256 weights
 */
function W_2_function_random_no() {
  let W_2_length = A_1_length * A_2_length;
  for (let i = 0; i < W_2_length; i++) {
    W_2.push(random(2));
  }
}

/**
 * Initialize biases for Layer 2
 * Creates one bias value per neuron in the second hidden layer
 */
function B_2_function_random_no() {
  let B_2_length = A_2_length;
  for (let i = 0; i < B_2_length; i++) {
    B_2.push(random(2));
  }
}

/**
 * Initialize weights for Layer 3 (Hidden2 -> Output)
 * Shape: A_3_length x A_2_length = 10 x 16 = 160 weights
 */
function W_3_function_random_no() {
  let W_3_length = A_2_length * A_3_length;
  for (let i = 0; i < W_3_length; i++) {
    W_3.push(random(2));
  }
}

/**
 * Initialize biases for Layer 3 (Output layer)
 * Creates one bias value per output neuron (10 total for digits 0-9)
 */
function B_3_function_random_no() {
  let B_3_length = A_3_length;
  for (let i = 0; i < B_3_length; i++) {
    B_3.push(random(2));
  }
}

/**
 * NOTE: Weight/bias initialization is commented out because
 * pre-trained weights are loaded from data.js
 * Uncomment these to train from scratch with random initialization
 */
// W_1_function_random_no();
// B_1_function_random_no();
// W_2_function_random_no();
// B_2_function_random_no();
// W_3_function_random_no();
// B_3_function_random_no();

/* =============================================================================
 * BACKPROPAGATION GRADIENT VARIABLES
 * ============================================================================= */

/**
 * Gradient Variables for Backpropagation
 * 
 * Naming Convention:
 *   dZ = ∂L/∂z (gradient w.r.t. pre-activation)
 *   dW = ∂L/∂W (gradient w.r.t. weights)
 *   dB = ∂L/∂b (gradient w.r.t. biases)
 * 
 * These gradients are computed during backpropagation and used
 * to update parameters via gradient descent.
 */
let dZ3 = [];   // Output layer: ∂L/∂z³
let dW3 = [];   // Output layer: ∂L/∂W³
let dB3 = [];   // Output layer: ∂L/∂b³
let dZ2 = null; // Hidden layer 2: ∂L/∂z²
let dW2 = [];   // Hidden layer 2: ∂L/∂W²
let dB2 = [];   // Hidden layer 2: ∂L/∂b²
let dZ1 = null; // Hidden layer 1: ∂L/∂z¹
let dW1 = [];   // Hidden layer 1: ∂L/∂W¹
let dB1 = [];   // Hidden layer 1: ∂L/∂b¹

/** 
 * Number of training examples (m)
 * Used for averaging gradients across the batch
 */
let m = Object.keys(Neural_Network_Train_Data).length;

/** 
 * Training metrics string for graph visualization
 * Accumulates cost and accuracy values across iterations
 */
let graphdata = ``;

/* =============================================================================
 * TRAINING FUNCTION - BACKPROPAGATION & PARAMETER UPDATE
 * ============================================================================= */

/**
 * Main training function that performs one iteration of:
 * 1. Forward propagation through all training examples
 * 2. Backpropagation to compute gradients
 * 3. Parameter update using gradient descent
 */
function train_neural_network() {
  
  /**
   * Backpropagation Algorithm
   * Computes gradients for all layers by propagating the error
   * backwards from the output layer to the input layer
   */
  function back_propogation() {
    // Temporary variables for layer activations during forward pass
    let xA_0;           // Current input (pixel data)
    let xA_1;           // Current hidden layer 1 activation
    let xA_2;           // Current hidden layer 2 activation
    let xA_3;           // Current output layer activation (before softmax)
    
    // Arrays to store all activations across training examples (needed for gradient computation)
    let total_xA_0 = []; // All input activations
    let total_xA_1 = []; // All hidden layer 1 activations
    let total_xA_2 = []; // All hidden layer 2 activations
    
    let softmax_xA_3;    // Softmax output for gradient calculation
    let cost_data = [];  // Cross-entropy loss for each training example
    let accuracy = 0;    // Count of correct predictions
    
    /**
     * Forward pass through all training examples
     * For each example, compute activations and output layer gradient (dZ3)
     */
    for (const key in Neural_Network_Train_Data) {
      // Get input data (pixel values) for current training example
      xA_0 = Neural_Network_Train_Data[key][0];
      total_xA_0.push(xA_0);
      
      // Forward propagation: Input -> Hidden Layer 1
      xA_1 = forward_propogation(
        Neural_Network_Train_Data[key][0],
        W_1,
        B_1,
        "TanH"
      );
      total_xA_1.push(xA_1);
      
      // Forward propagation: Hidden Layer 1 -> Hidden Layer 2
      xA_2 = forward_propogation(xA_1, W_2, B_2, "TanH");
      total_xA_2.push(xA_2);
      
      // Forward propagation: Hidden Layer 2 -> Output Layer
      xA_3 = forward_propogation(xA_2, W_3, B_3, "TanH");
      
      // Apply softmax to get probability distribution
      softmax_xA_3 = softmax(xA_3);
      let othersoftmax = softmax(xA_3); // Copy for accuracy calculation
      
      // Compute output layer gradient: dZ3 = predicted - actual (one-hot encoded)
      // Subtracting 1 from the true class creates the gradient for cross-entropy loss
      softmax_xA_3[Neural_Network_Train_Data[key][1]] =
        softmax_xA_3[Neural_Network_Train_Data[key][1]] - 1;
      dZ3.push(softmax_xA_3);

      // Calculate cross-entropy loss for this example: -log(p_correct)
      cost_data.push(log(othersoftmax[Neural_Network_Train_Data[key][1]] * 1));
      
      // Check if prediction matches the true label (for accuracy calculation)
      if (othersoftmax[Neural_Network_Train_Data[key][1]] == Math.max.apply(null, othersoftmax)) {
        accuracy++;
      }
    }

    /**
     * Calculate average cross-entropy cost across all training examples
     * Cost = -(1/m) * Σ log(p_correct) where p_correct is the predicted probability
     * of the true class
     */
    let new_cost = 0;
    for (let gh = 0; gh < cost_data.length; gh++) {
      new_cost = new_cost + cost_data[gh];
    }
    new_cost = -(new_cost / m);

    // Log training metrics to console and UI
    console.log("The cost is " + new_cost);
    console.log("The accuracy is " + (accuracy/m)*100);
    cost_value_box.innerHTML = "The Cost Is " + new_cost + "<br>" + "The Accuracy Is " + (accuracy/m)*100;
    
    // Append metrics to graph data string for visualization
    graphdata += `the cost is ` + new_cost + `\n` + `The accuracy is ` + (accuracy/m)*100 + `\n`;
    graphgenerater();

    /**
     * Compute dW3: Gradient of cost with respect to weights in layer 3
     * Formula: dW3 = (1/m) * dZ3^T * A_2
     * This tells us how to adjust W3 to minimize the cost
     */
    let multiply_dZ3_xA_2T;
    multiply_dZ3_xA_2T = matrix_multipilcation_with_transpose(
      dZ3,
      transposeMatrix(total_xA_2)
    );

    // Average the gradients across all training examples
    for (let r = 0; r < multiply_dZ3_xA_2T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ3_xA_2T[r].length; l++) {
        divide_by_r.push(multiply_dZ3_xA_2T[r][l] / m);
      }
      dW3.push(divide_by_r);
    }

    /**
     * Compute dB3: Gradient of cost with respect to biases in layer 3
     * Formula: dB3 = (1/m) * Σ dZ3
     * Sum along the example axis and average
     */
    let transpose_dZ3 = transposeMatrix(dZ3);
    for (let v = 0; v < transpose_dZ3.length; v++) {
      let sum = 0;
      sum = transpose_dZ3[v].reduce((accumulator, currentValue) => {
        return accumulator + currentValue;
      }, 0);
      dB3.push(sum);
    }
    let xxdB3 = dB3.map((num) => num / m);
    dB3 = xxdB3;
    
    /**
     * Compute dZ2: Gradient of cost with respect to hidden layer 2 pre-activation
     * Formula: dZ2 = (W3^T * dZ3) ⊙ g'(Z2)
     * Where ⊙ is element-wise multiplication and g'(Z2) is the derivative of TanH
     */
    
    // Convert W_3 flat array to matrix form for transpose
    let matrix_of_W_3 = [];
    let g_sum = [];
    for (let g = 0; g <= W_3.length; g++) {
      if (g_sum.length === A_2_length) {
        matrix_of_W_3.push(g_sum);
        g_sum = [];
      }
      g_sum.push(W_3[g]);
    }
    matrix_of_W_3 = transposeMatrix(matrix_of_W_3);
    
    // Multiply W3 transpose by dZ3
    let multiply_of_W3T_dZ3 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_3),
      dZ3
    );
    
    // Compute derivative of TanH for hidden layer 2
    let A_2_derivative = derivative_tanH(total_xA_2);
    
    // Element-wise multiplication to get dZ2
    dZ2 = element_wise_multiplication(multiply_of_W3T_dZ3, A_2_derivative);

    /**
     * Compute dW2: Gradient of cost with respect to weights in layer 2
     * Formula: dW2 = (1/m) * dZ2^T * A_1
     */
    let multiply_dZ2_xA_2T;
    multiply_dZ2_xA_2T = matrix_multipilcation_with_transpose(
      dZ2,
      transposeMatrix(total_xA_1)
    );
    for (let r = 0; r < multiply_dZ2_xA_2T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ2_xA_2T[r].length; l++) {
        divide_by_r.push(multiply_dZ2_xA_2T[r][l] / m);
      }
      dW2.push(divide_by_r);
    }

    /**
     * Compute dB2: Gradient of cost with respect to biases in layer 2
     * Formula: dB2 = (1/m) * Σ dZ2
     */
    let transpose_dZ2 = transposeMatrix(dZ2);
    for (let v = 0; v < transpose_dZ2.length; v++) {
      let sum = 0;
      sum = transpose_dZ2[v].reduce((accumulator, currentValue) => {
        return accumulator + currentValue;
      }, 0);
      dB2.push(sum);
    }
    let xxdB2 = dB2.map((num) => num / m);
    dB2 = xxdB2;
    
    /**
     * Compute dZ1: Gradient of cost with respect to hidden layer 1 pre-activation
     * Formula: dZ1 = (W2^T * dZ2) ⊙ g'(Z1)
     * This continues the chain rule backwards through the network
     */
    
    // Convert W_2 flat array to matrix form for transpose
    let matrix_of_W_2 = [];
    let q_sum = [];
    for (let g = 0; g <= W_2.length; g++) {
      if (q_sum.length === A_1_length) {
        matrix_of_W_2.push(q_sum);
        q_sum = [];
      }
      q_sum.push(W_2[g]);
    }
    matrix_of_W_2 = transposeMatrix(matrix_of_W_2);
    
    // Multiply W2 transpose by dZ2
    let multiply_of_W2T_dZ2 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_2),
      dZ2
    );
    
    // Compute derivative of TanH for hidden layer 1
    let A_1_derivative = derivative_tanH(total_xA_1);
    
    // Element-wise multiplication to get dZ1
    dZ1 = element_wise_multiplication(multiply_of_W2T_dZ2, A_1_derivative);

    /**
     * Compute dW1: Gradient of cost with respect to weights in layer 1 (input layer)
     * Formula: dW1 = (1/m) * dZ1^T * A_0 (input)
     */
    let multiply_dZ1_xA_1T;
    multiply_dZ1_xA_1T = matrix_multipilcation_with_transpose(
      dZ1,
      transposeMatrix(total_xA_0)
    );
    for (let r = 0; r < multiply_dZ1_xA_1T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ1_xA_1T[r].length; l++) {
        divide_by_r.push(multiply_dZ1_xA_1T[r][l] / m);
      }
      dW1.push(divide_by_r);
    }

    /**
     * Compute dB1: Gradient of cost with respect to biases in layer 1
     * Formula: dB1 = (1/m) * Σ dZ1
     */
    let transpose_dZ1 = transposeMatrix(dZ1);
    for (let v = 0; v < transpose_dZ1.length; v++) {
      let sum = 0;
      sum = transpose_dZ1[v].reduce((accumulator, currentValue) => {
        return accumulator + currentValue;
      }, 0);
      dB1.push(sum);
    }
    let xxdB1 = dB1.map((num) => num / m);
    dB1 = xxdB1;

    // End of backpropagation - all gradients computed
  }

  // Execute backpropagation
  back_propogation();

  /**
   * Update network parameters using gradient descent
   * New parameter = Old parameter - (learning_rate * gradient)
   * This moves the parameters in the direction that reduces the cost
   */
  function update_parameters() {
    W_1 = W_update(W_1, alpha, dW1);  // Update layer 1 weights
    W_2 = W_update(W_2, alpha, dW2);  // Update layer 2 weights
    W_3 = W_update(W_3, alpha, dW3);  // Update layer 3 weights
    B_1 = B_update(B_1, alpha, dB1);  // Update layer 1 biases
    B_2 = B_update(B_2, alpha, dB2);  // Update layer 2 biases
    B_3 = B_update(B_3, alpha, dB3);  // Update layer 3 biases
  }
  
  update_parameters();
}

/* =============================================================================
 * PREDICTION FUNCTION - USER INPUT PROCESSING
 * ============================================================================= */

/**
 * Main function for digit recognition
 * Called when user clicks "Recognize" button
 * Performs forward propagation on user-drawn input and displays prediction
 */
let newinterval = null;  // Interval reference (legacy, kept for compatibility)

function neural_network_main() {
  // Stop any previous prediction interval
  if (newinterval) {
    clearInterval(newinterval);
    newinterval = null;
  }
  
  // Convert the pixel grid to input array (0 = white, 1 = black)
  button.forEach((btn) => {
    if (btn.style.background != "black") {
      A_0.push(0);  // White pixel = 0
    } else {
      A_0.push(1);  // Black pixel = 1
    }
  });

  // Forward propagation through the network
  A_1 = forward_propogation(A_0, W_1, B_1, "TanH");  // Input -> Hidden 1
  A_2 = forward_propogation(A_1, W_2, B_2, "TanH");  // Hidden 1 -> Hidden 2
  A_3 = forward_propogation(A_2, W_3, B_3, "TanH");  // Hidden 2 -> Output

  // Apply softmax to get probability distribution
  const softmax_output = softmax(A_3);
  console.log(softmax_output);

  // Find and display the predicted digit (highest probability)
  const maxProb = Math.max(...softmax_output);
  const predictedDigit = softmax_output.indexOf(maxProb);
  output_text.innerHTML = `<strong>Predicted: ${predictedDigit}</strong>`;

  // Display all digits with their confidence percentages (sorted by probability)
  const percentageOutput = convertToPercentages(softmax_output);
  for (const item of percentageOutput) {
    const percent = item.percentage.toFixed(2);
    output_text.innerHTML += `<br>${item.index} = ${percent}%`;
  }

  // Reset activation arrays for next prediction
  A_0 = [];
  A_1 = [];
  A_2 = [];
  A_3 = [];
}



/* =============================================================================
 * DATA AUGMENTATION FUNCTIONS
 * These functions modify training data to improve model generalization
 * ============================================================================= */

/**
 * Adds random noise to a binary image
 * 
 * Randomly converts white pixels (0) to black (1) based on probability.
 * Black pixels remain unchanged (preserves the drawn digit).
 * 
 * @param {number[]} image - Flattened binary pixel array
 * @param {number} width - Image width in pixels
 * @param {number} height - Image height in pixels  
 * @param {number} keepWhiteProb - Probability of keeping white pixel white (0-1)
 *                                 0.99 = 1% chance of noise per white pixel
 * @returns {number[]} Image with random noise added
 * 
 * @example
 * noisefunction(image, 20, 20, 0.99)  // 1% noise probability
 */
function noisefunction(image, width, height, keepWhiteProb) {
    return image.map(pixel => {
        if (pixel === 0) {
            // White pixel: randomly flip to black based on probability
            return Math.random() < keepWhiteProb ? 0 : 1;
        }
        // Black pixel: keep unchanged
        return 1;
    });
}

/**
 * Translates (shifts) an image by specified pixel amounts
 * 
 * Shifts all pixels by (dx, dy). Pixels that move outside the
 * image boundary are replaced with 0 (white/background).
 * 
 * @param {number[]} image - Flattened pixel array
 * @param {number} height - Image height
 * @param {number} width - Image width
 * @param {number} dx - Horizontal shift (+ = right, - = left)
 * @param {number} dy - Vertical shift (+ = down, - = up)
 * @returns {number[]} Translated image as flattened array
 * 
 * @example
 * position(image, 20, 20, 2, -1)  // Shift right 2px, up 1px
 */
function position(image, height, width, dx, dy) {
    // Convert flat array to 2D grid for easier manipulation
    const grid = [];
    for (let row = 0; row < height; row++) {
        grid.push(image.slice(row * width, (row + 1) * width));
    }
    
    // Create translated output grid
    const translated = [];
    
    for (let y = 0; y < height; y++) {
        translated[y] = [];
        for (let x = 0; x < width; x++) {
            // Calculate source coordinates
            const srcX = x + dx;
            const srcY = y + dy;
            
            // Check bounds and copy pixel or fill with white
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                translated[y][x] = grid[srcY][srcX];
            } else {
                translated[y][x] = 0;  // Out of bounds = white
            }
        }
    }
    
    return translated.flat();
}

/**
 * Rotates an image around its center
 * 
 * Uses 2D rotation matrix transformation:
 *   [x']   [cosθ  -sinθ] [x - cx]   [cx]
 *   [y'] = [sinθ   cosθ] [y - cy] + [cy]
 * 
 * @param {number[]} image - Flattened pixel array
 * @param {number} angleDegrees - Rotation angle in degrees (+ = counterclockwise)
 * @param {number} height - Image height
 * @param {number} width - Image width
 * @returns {number[]} Rotated image as flattened array
 * 
 * @example
 * rotation(image, 15, 20, 20)   // Rotate 15° counterclockwise
 * rotation(image, -10, 20, 20)  // Rotate 10° clockwise
 */
function rotation(image, angleDegrees, height, width) {
    // Convert degrees to radians
    const angleRad = (angleDegrees * Math.PI) / 180;
    const cos = Math.cos(angleRad);
    const sin = Math.sin(angleRad);
    
    // Calculate center point (rotation pivot)
    const centerX = width / 2;
    const centerY = height / 2;
    
    const rotated = new Array(width * height);
    
    // For each output pixel, find the corresponding source pixel
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Translate to origin (center)
            const relX = x - centerX;
            const relY = y - centerY;
            
            // Apply inverse rotation to find source coordinates
            // (we're mapping destination to source)
            const srcX = Math.round(relX * cos - relY * sin + centerX);
            const srcY = Math.round(relX * sin + relY * cos + centerY);
            
            // Sample from source or use background (0)
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                rotated[y * width + x] = image[srcY * width + srcX];
            } else {
                rotated[y * width + x] = 0;  // Background
            }
        }
    }
    
    return rotated;
}

/**
 * @deprecated Use randomrangenumber() from maths.js instead
 * Kept for backward compatibility
 */

/**
 * Applies random data augmentation to training data
 * 
 * Augmentation Pipeline:
 *   1. Random translation: ±4 pixels in x and y
 *   2. Random noise: 1% probability per white pixel
 *   3. Random rotation: ±15 degrees
 *   4. Cleanup: Replace undefined values with 0
 * 
 * This helps the model generalize better by seeing varied
 * versions of the training examples.
 * 
 * @param {Object} targetData - Training data object to modify
 * @param {Object} sourceData - Original training data (reference)
 */
function randomnessadder(targetData, sourceData) {
    const gridSize = NETWORK_CONFIG.GRID_WIDTH;
    
    for (const key in sourceData) {
        let augmented = sourceData[key][0];
        
        // Step 1: Random translation (±4 pixels)
        const dx = randomrangenumber(-4, 4);
        const dy = randomrangenumber(-4, 4);
        augmented = position(augmented, gridSize, gridSize, dx, dy);
        
        // Step 2: Add random noise (1% probability)
        augmented = noisefunction(augmented, gridSize, gridSize, 0.99);
        
        // Step 3: Random rotation (±15 degrees)
        const angle = randomrangenumber(-15, 15);
        augmented = rotation(augmented, angle, gridSize, gridSize);
        
        // Step 4: Cleanup - replace null/undefined with 0
        augmented = augmented.map(val => (val == null) ? 0 : val);
        
        // Update training data with augmented version
        targetData[key][0] = augmented;
    }
}

/** Deep copy of training data (reference for augmentation) */
const neuralnetworkdatacopy = JSON.parse(JSON.stringify(Neural_Network_Train_Data));

/* =============================================================================
 * TRAINING CONFIGURATION & UI CONTROLS
 * ============================================================================= */

/**
 * Training Configuration
 */
let training_length = 20;           // Number of training iterations
let loadpercent = 0;                // Progress bar percentage
let useDataAugmentation = true;     // Enable/disable data augmentation
let train_interval = null;          // Reference to training interval

// Initialize loading bar
loadbar.style.width = `calc(90% / ${training_length} * ${loadpercent})`;

/**
 * Main Training Function
 * 
 * Executes the training loop:
 *   1. Optionally augment training data
 *   2. Forward propagation (compute predictions)
 *   3. Backpropagation (compute gradients)
 *   4. Parameter update (gradient descent)
 *   5. Reset gradients for next iteration
 * 
 * Updates the progress bar and displays metrics during training.
 * Uses async iteration to prevent UI blocking.
 */
let isTraining = false;  // Prevent multiple simultaneous training sessions

function train_button() {
    // Prevent starting new training while already training
    if (isTraining) {
        console.log("Training already in progress...");
        return;
    }
    
    isTraining = true;
    loadpercent = 0;
    
    // Show progress bar
    loadbarcontain.style.display = "flex";
    loadbar.style.width = "0%";
    
    /**
     * Async training iteration
     * Uses setTimeout to yield control back to browser for UI updates
     */
    function trainIteration(epoch) {
        if (epoch >= training_length) {
            // Training complete
            loadbarcontain.style.display = "none";
            loadpercent = 0;
            isTraining = false;
            console.log("Training complete!");
            return;
        }
        
        // Step 1: Apply data augmentation if enabled
        if (useDataAugmentation) {
            randomnessadder(Neural_Network_Train_Data, neuralnetworkdatacopy);
        }
        
        // Step 2-4: Forward prop, backprop, and parameter update
        train_neural_network();
        
        // Step 5: Reset gradients for next iteration
        // (Critical: gradients accumulate and must be cleared)
        dZ3 = [];
        dW3 = [];
        dB3 = [];
        dZ2 = null;
        dW2 = [];
        dB2 = [];
        dZ1 = null;
        dW1 = [];
        dB1 = [];
        
        // Update progress bar
        loadpercent = epoch + 1;
        const progressPercent = (loadpercent / training_length) * 90;
        loadbar.style.width = `${progressPercent}%`;
        
        // Schedule next iteration (setTimeout allows browser to repaint)
        setTimeout(() => trainIteration(epoch + 1), 0);
    }
    
    // Start training
    trainIteration(0);
}

/**
 * Hyperparameter Update Handler
 * 
 * Allows users to modify training parameters without reloading:
 *   - Learning rate (α): Controls gradient descent step size
 *   - Regularization (λ): L2 penalty coefficient  
 *   - Training iterations: Number of epochs per train button click
 */
alpha_submit.addEventListener("click", () => {
    alpha = parseFloat(alpha_value.value) || 0.1;
    lambda = parseFloat(document.getElementById("lambda_value").value) || 0.000001;
    training_length = parseInt(train_length_input.value, 10) || 20;
    
    console.log(`Hyperparameters updated: α=${alpha}, λ=${lambda}, iterations=${training_length}`);
});

/**
 * Event listener for data augmentation toggle
 * Enables/disables random transformations (rotation, translation, noise) during training
 */
const augmentationToggle = document.getElementById("augmentation_toggle");
const augmentationStatus = document.getElementById("augmentation_status");

if (augmentationToggle) {
  augmentationToggle.addEventListener("change", () => {
    useDataAugmentation = augmentationToggle.checked;
    if (augmentationStatus) {
      augmentationStatus.textContent = useDataAugmentation ? "Enabled" : "Disabled";
      augmentationStatus.classList.toggle("active", useDataAugmentation);
    }
    console.log("Data Augmentation: " + (useDataAugmentation ? "Enabled" : "Disabled"));
  });
}