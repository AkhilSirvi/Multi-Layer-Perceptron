/**
 * Multi-Layer Perceptron (MLP) Neural Network for Handwritten Digit Recognition
 * 
 * This script implements a 3-layer neural network that can:
 * 1. Recognize handwritten digits (0-9) drawn by the user
 * 2. Train on the MNIST-style dataset using backpropagation
 * 3. Display training progress with cost and accuracy metrics
 * 
 * Network Architecture:
 * - Input Layer (A_0): 400 neurons (20x20 pixel grid)
 * - Hidden Layer 1 (A_1): 16 neurons with TanH activation
 * - Hidden Layer 2 (A_2): 16 neurons with TanH activation
 * - Output Layer (A_3): 10 neurons (digits 0-9) with Softmax activation
 * 
 * @author Akhil Sirvi
 */

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
 * Creates a 20x20 pixel grid (400 pixels) for drawing digits
 * Each pixel is a clickable/hoverable div element that can be toggled to black
 * This forms the input layer of the neural network
 */
function pixel_creater() {
  for (let pixel_creater_i = 0; pixel_creater_i < 400; pixel_creater_i++) {
    // Create a new div element for each pixel
    let pixel = document.createElement("div");
    second_box.appendChild(pixel);
    
    // Generate unique IDs for row/column tracking
    let pixel_creater_v = 1;
    let pixel_creater_k = 1;
    pixel_creater_v++;
    if (pixel_creater_v == 10) {
      pixel_creater_k++;
      pixel_creater_v = 0;
    }
    
    // Assign ID, class, and disable default drag behavior
    pixel.id = pixel_creater_v + "rowbox" + pixel_creater_k;
    pixel.className = "box";
    pixel.draggable = false;
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
  // Desktop: Draw when hovering with mouse button pressed
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
 * NEURAL NETWORK ARCHITECTURE CONFIGURATION
 * ============================================================================= */

/**
 * Layer dimensions define the network architecture:
 * - A_0 (Input):   400 neurons - Represents 20x20 pixel grid (flattened)
 * - A_1 (Hidden1): 16 neurons  - First hidden layer with TanH activation
 * - A_2 (Hidden2): 16 neurons  - Second hidden layer with TanH activation  
 * - A_3 (Output):  10 neurons  - One neuron per digit (0-9) with Softmax
 */
let A_0_length = 400;  // Input layer: 20x20 = 400 pixels
let A_1_length = 16;   // First hidden layer neurons
let A_2_length = 16;   // Second hidden layer neurons
let A_3_length = 10;   // Output layer: 10 digit classes (0-9)

/** Activation arrays for each layer (populated during forward propagation) */
let A_0 = [];  // Input activations (pixel values: 0 or 1)
let A_1 = [];  // First hidden layer activations
let A_2 = [];  // Second hidden layer activations
let A_3 = [];  // Output layer activations (before softmax)

/**
 * Learning rate (alpha) - Controls the step size during gradient descent
 * Smaller values = slower but more stable learning
 * Larger values = faster but potentially unstable learning
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
 * Gradient variables for backpropagation:
 * dZ = Gradient of cost with respect to the pre-activation (z)
 * dW = Gradient of cost with respect to weights
 * dB = Gradient of cost with respect to biases
 * 
 * These gradients are computed during backpropagation and used to update
 * the network's weights and biases using gradient descent.
 */
let dZ3 = [];   // Output layer gradient
let dW3 = [];   // Output layer weight gradient
let dB3 = [];   // Output layer bias gradient
let dZ2 = null; // Second hidden layer gradient
let dW2 = [];   // Second hidden layer weight gradient
let dB2 = [];   // Second hidden layer bias gradient
let dZ1 = null; // First hidden layer gradient
let dW1 = [];   // First hidden layer weight gradient
let dB1 = [];   // First hidden layer bias gradient

/** Number of training examples (m) - used for averaging gradients */
let m = Object.keys(Neural_Network_Train_Data).length;

/** String to accumulate training metrics for graph visualization */
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
      othersoftmax = softmax(xA_3); // Copy for accuracy calculation
      
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
 * Called when user clicks "Enter your written data" button
 * Performs forward propagation on user-drawn input and displays prediction
 */
function neural_network_main() {
  // Set up interval to continuously process drawing (allows real-time updates)
  newinterval = setInterval(() => {
    // Convert the pixel grid to input array (0 = white, 1 = black)
    button.forEach((button) => {
      if (button.style.background != "black") {
        A_0.push(0);  // White pixel = 0
      } else {
        A_0.push(1);  // Black pixel = 1
      }
    });

    // Forward propagation through the network
    A_1 = forward_propogation(A_0, W_1, B_1, "TanH");  // Input -> Hidden 1
    A_2 = forward_propogation(A_1, W_2, B_2, "TanH");  // Hidden 1 -> Hidden 2
    A_3 = forward_propogation(A_2, W_3, B_3, "TanH");  // Hidden 2 -> Output

    console.log(softmax(A_3));

    // Find and display the predicted digit (highest probability)
    let softmax_output = softmax(A_3);
    for (let j = 0; j < softmax_output.length; j++) {
      if (softmax_output[j] == Math.max.apply(null, softmax_output)) {
        output_text.innerHTML = "The output is " + j;
      }
    }

    // Display all digits with their confidence percentages (sorted by probability)
    let percentageoutput = convertToPercentages(softmax(A_3));
    let percentageoutput_index = percentageoutput.map((item) => item.index);
    let percentageoutput_percent = percentageoutput.map(
      (item) => item.percentage
    );
    for (let u = 0; u < percentageoutput.length; u++) {
      output_text.innerHTML +=
        "<br>" + percentageoutput_index[u] + "=" + percentageoutput_percent[u];
    }

    // Reset activation arrays for next iteration
    A_0 = [];
    A_1 = [];
    A_2 = [];
    A_3 = [];
  }, 100);  // Update every 100ms
  
  // Clear the drawing grid when starting recognition
  button.forEach((button) => {
    button.style.background = `white`;
  });
}



/* =============================================================================
 * DATA AUGMENTATION FUNCTIONS
 * These functions modify training data to improve model generalization
 * ============================================================================= */

/**
 * Adds random noise to a binary image array
 * Randomly flips white pixels (0) to black (1) based on bias factor
 * 
 * @param {Array} inputarray - Flattened binary pixel array
 * @param {number} width - Image width (pixels)
 * @param {number} height - Image height (pixels)
 * @param {number} biasFactor - Probability of keeping a white pixel white (0-1)
 *                              Higher values = less noise added
 * @returns {Array} Noised image array
 */
function noisefunction(inputarray, width, height, biasFactor) {
  let noisedarray = [];

  for (let noise_loop = 0; noise_loop < width*height; noise_loop++) {
    if (inputarray[noise_loop] === 0) {
      // For white pixels: randomly decide to keep white or flip to black
      noisedarray.push(Math.random() < biasFactor ? 0 : 1);
    }
    else {
      // Keep black pixels unchanged
      noisedarray.push(1);
    }
  }

  return noisedarray;
}

/**
 * Translates (shifts) an image by dx and dy pixels
 * Pixels that shift outside the boundary are replaced with 0 (white)
 * 
 * @param {Array} inputarray - Flattened pixel array
 * @param {number} height - Image height
 * @param {number} width - Image width
 * @param {number} dxv - Horizontal shift (positive = right, negative = left)
 * @param {number} dyv - Vertical shift (positive = down, negative = up)
 * @returns {Array} Translated image as flattened array
 */
function position(inputarray, height, width, dxv, dyv) {
  // Convert flattened array to 2D grid
  const originalArray = Array.from({ length: height }, (_, i) => inputarray.slice(i * width, (i + 1) * width));
  const dx = dxv;
  const dy = dyv;
  
  const rows = originalArray.length;
  const columns = originalArray[0].length;

  const translatedArray = [];

  // Apply translation transformation to each pixel
  for (let i = 0; i < rows; i++) {
    translatedArray[i] = [];
    for (let j = 0; j < columns; j++) {
      const originalX = j;
      const originalY = i;
      const translatedX = originalX + dx;  // New X position after shift
      const translatedY = originalY + dy;  // New Y position after shift
      
      // Check if the source position is within bounds
      if (translatedX >= 0 && translatedX < columns && translatedY >= 0 && translatedY < rows) {
        translatedArray[i][j] = originalArray[translatedY][translatedX];
      } 
      else {
        // Fill out-of-bounds pixels with white (0)
        translatedArray[i][j] = 0;
      }
    }
  }
  return translatedArray.flat();  // Return as flattened 1D array
}

/**
 * Rotates an image around its center by a specified angle
 * Uses 2D rotation matrix transformation
 * 
 * @param {Array} inputarray - Flattened pixel array
 * @param {number} anglevalue - Rotation angle in degrees
 * @param {number} height - Image height
 * @param {number} width - Image width  
 * @returns {Array} Rotated image as flattened array
 */
function rotation(inputarray, anglevalue, height, width) {
  // Convert degrees to radians for Math functions
  let angle = (anglevalue * Math.PI) / 180;
  
  // Calculate center point for rotation pivot
  const centerX = width / 2;
  const centerY = height / 2;

  const rotatedarray = [];

  // Apply rotation transformation to each pixel
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Calculate position relative to center
      const relX = x - centerX;
      const relY = y - centerY;
        
      // Apply 2D rotation matrix: [cosθ -sinθ] [x]
      //                           [sinθ  cosθ] [y]
      const newX = Math.round(relX * Math.cos(angle) - relY * Math.sin(angle));
      const newY = Math.round(relX * Math.sin(angle) + relY * Math.cos(angle));

      // Convert back to absolute coordinates
      const absX = Math.floor(newX + centerX);
      const absY = Math.floor(newY + centerY);
  
      // Get the pixel value from the source position
      const pixel = inputarray[absY * width + absX];
  
      // Assign to the rotated position
      rotatedarray[y * width + x] = pixel;
    }
  }
  
  return rotatedarray;
}

/**
 * Generates a random integer between min and max (inclusive)
 * 
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Random integer in range [min, max]
 */
function randomrangenumber(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Applies random data augmentation to training data
 * Augmentations include: translation, noise, and rotation
 * This helps the model generalize better by seeing varied versions of training examples
 * 
 * @param {Object} originaldata - Training data object to be modified
 * @param {Object} copydata - Original copy of training data for reference
 */
function randomnessadder(originaldata, copydata) {
  for (const key in copydata) {
    // Step 1: Random translation (-4 to +4 pixels in each direction)
    let noisedarray1 = position(copydata[key][0], 20, 20, randomrangenumber(-4, 4), randomrangenumber(-4, 4));
    
    // Step 2: Add random noise (99% chance to keep white pixels white)
    let noisedarray2 = noisefunction(noisedarray1, 20, 20, 0.99);
    
    // Step 3: Random rotation (-15 to +15 degrees)
    let noisedarray3 = rotation(noisedarray2, randomrangenumber(-15, 15), 20, 20);
    
    // Step 4: Replace null/undefined values with 0 (cleanup)
    let noisedarray4 = noisedarray3.map((value) => {
      if (value === null || value === undefined) {
        return 0;
      } else {
        return value;
      }
    });
    
    // Update the training data with augmented version
    originaldata[key][0] = noisedarray4;
  }
}

/** Deep copy of original training data (used as reference for augmentation) */
const neuralnetworkdatacopy = JSON.parse(JSON.stringify(Neural_Network_Train_Data));;

/* =============================================================================
 * TRAINING LOOP & UI CONTROLS
 * ============================================================================= */

/** Number of training iterations to perform */
let training_length = 20;

/** Current progress percentage for the loading bar */
let loadpercent = 0;

// Initialize loading bar width
loadbar.style.width = "calc(90%/" + training_length + "*" + loadpercent + ")";

/** Reference to training interval (if using async training) */
let train_interval = null;

/**
 * Initiates the training process when user clicks "Train the neural network" button
 * Runs multiple training iterations with data augmentation
 * Displays progress via loading bar
 */
function train_button() {
  // Show the loading bar
  loadbarcontain.style.display = "flex";
  
  // Training loop - run for specified number of iterations
  for (let trainingloop = 0; trainingloop < training_length; trainingloop++) {
    // Apply random augmentation to training data for each iteration
    randomnessadder(Neural_Network_Train_Data, neuralnetworkdatacopy);
    
    // Run one training iteration (forward prop + backprop + parameter update)
    train_neural_network();
    
    // Reset gradient arrays for next iteration
    // (Gradients must be cleared between training iterations)
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
    loadpercent++;
    loadbar.style.width = "calc(90%/" + training_length + "*" + loadpercent + ")";
    
    // Hide loading bar when training is complete
    if (training_length == loadpercent) {
      loadbarcontain.style.display = "none";
      loadpercent = 0;
    }
  }
}

/**
 * Event listener for hyperparameter update button
 * Allows users to modify training parameters without reloading the page
 * Updates: learning rate (alpha), regularization (lambda), and training iterations
 */
alpha_submit.addEventListener("click", () => {
  alpha = parseFloat(alpha_value.value);  // Update learning rate
  lambda = parseFloat(document.getElementById("lambda_value").value);  // Update regularization
  training_length = parseFloat(train_length_input.value);  // Update number of training iterations
});