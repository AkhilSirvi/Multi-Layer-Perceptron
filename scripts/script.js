"use strict";

const NETWORK_CONFIG = Object.freeze({
  INPUT_SIZE: 784, // 28×28 pixel grid
  HIDDEN_1_SIZE: 128, // First hidden layer neurons
  HIDDEN_2_SIZE: 64, // Second hidden layer neurons
  OUTPUT_SIZE: 10, // Digit classes (0-9)
  GRID_WIDTH: 28, // Drawing grid width
  GRID_HEIGHT: 28, // Drawing grid height
});

let second_box = document.getElementById("second");
let output_text = document.getElementById("output_text");
let loadbar = document.getElementById("loadbarmain");
let loadbarcontain = document.getElementById("loadbar");
let cost_value_box = document.getElementById("cost_value_output");
let alpha_value = document.getElementById("alpha_value");
let train_length_input = document.getElementById("train_length_input");
let alpha_submit = document.getElementById("alpha_submit");

function pixel_creater() {
  const totalPixels = NETWORK_CONFIG.GRID_WIDTH * NETWORK_CONFIG.GRID_HEIGHT;

  for (let i = 0; i < totalPixels; i++) {
    const pixel = document.createElement("div");
    pixel.className = "box";
    pixel.draggable = false;
    pixel.dataset.index = i;
    pixel.dataset.intensity = 0;
    second_box.appendChild(pixel);
  }
}

pixel_creater();

let body = document.body;
let button = document.querySelectorAll(".box");
let click = false;

function paintWithThickness(index) {
  const gridWidth = NETWORK_CONFIG.GRID_WIDTH;
  const gridHeight = NETWORK_CONFIG.GRID_HEIGHT;
  const row = Math.floor(index / gridWidth);
  const col = index % gridWidth;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      const newRow = row + dr;
      const newCol = col + dc;
      if (
        newRow >= 0 &&
        newRow < gridHeight &&
        newCol >= 0 &&
        newCol < gridWidth
      ) {
        const neighborIndex = newRow * gridWidth + newCol;
        const pixel = button[neighborIndex];
        const distance = Math.abs(dr) + Math.abs(dc);
        const intensity = distance === 0 ? 1 : 1 * 0.3;
        const currentIntensity = parseFloat(pixel.dataset.intensity) || 0;
        const newIntensity = Math.min(1.0, currentIntensity + intensity);
        pixel.dataset.intensity = newIntensity;
        const grayValue = Math.round((1 - newIntensity) * 255);
        pixel.style.background = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
      }
    }
  }
}

body.addEventListener("mousedown", (event) => {
  if (event.buttons === 1) {
    click = true;
  }
});

body.addEventListener("mouseup", (event) => {
  if (event.button === 0) {
    click = false;
  }
});

button.forEach((btn, index) => {
  btn.addEventListener("mousedown", (event) => {
    if (event.buttons === 1) {
      paintWithThickness(index);
    }
  });

  btn.addEventListener("mouseover", () => {
    if (click == true) {
      paintWithThickness(index);
    }
  });

  second_box.addEventListener("touchmove", (event) => {
    event.preventDefault();
    const touch = event.touches[0];
    const x = touch.clientX;
    const y = touch.clientY;
    const rect = btn.getBoundingClientRect();

    if (
      x >= rect.left &&
      x <= rect.right &&
      y >= rect.top &&
      y <= rect.bottom
    ) {
      paintWithThickness(index);
    }
  });
});

let A_0_length = NETWORK_CONFIG.INPUT_SIZE;
let A_1_length = NETWORK_CONFIG.HIDDEN_1_SIZE;
let A_2_length = NETWORK_CONFIG.HIDDEN_2_SIZE;
let A_3_length = NETWORK_CONFIG.OUTPUT_SIZE;

let A_0 = []; // Input layer activations
let A_1 = []; // Hidden layer 1 activations (after activation function)
let A_2 = []; // Hidden layer 2 activations (after activation function)
let A_3 = []; // Output layer activations (before softmax)

// learing rate
let alpha = 0.1;

function randomArray(length) {
  return Array.from({ length }, () => random(2));
}
// W_1 = []; // Weights for layer 1 (input -> hidden 1)
// B_1 = []; // Biases for layer 1
// W_2 = []; // Weights for layer 2 (hidden 1 -> hidden 2)
// B_2 = []; // Biases for layer 2
// W_3 = []; // Weights for layer 3 (hidden 2 -> output)
// B_3 = []; // Biases for layer 3

function W_1_function_random_no() {
  W_1 = randomArray(A_1_length * A_0_length);
}
function B_1_function_random_no() {
  B_1 = randomArray(A_1_length);
}
function W_2_function_random_no() {
  W_2 = randomArray(A_1_length * A_2_length);
}
function B_2_function_random_no() {
  B_2 = randomArray(A_2_length);
}
function W_3_function_random_no() {
  W_3 = randomArray(A_2_length * A_3_length);
}
function B_3_function_random_no() {
  B_3 = randomArray(A_3_length);
}

if (typeof W_1 === "undefined" || W_1.length === 0) {
  console.log("Initializing weights from scratch...");
  W_1_function_random_no();
  B_1_function_random_no();
  W_2_function_random_no();
  B_2_function_random_no();
  W_3_function_random_no();
  B_3_function_random_no();
} else {
  console.log("Pre-trained weights loaded from data.js");
  console.log(
    `W_1: ${W_1.length} params (expected: ${NETWORK_CONFIG.INPUT_SIZE * NETWORK_CONFIG.HIDDEN_1_SIZE})`,
  );
  console.log(
    `W_2: ${W_2.length} params (expected: ${NETWORK_CONFIG.HIDDEN_1_SIZE * NETWORK_CONFIG.HIDDEN_2_SIZE})`,
  );
  console.log(
    `W_3: ${W_3.length} params (expected: ${NETWORK_CONFIG.HIDDEN_2_SIZE * NETWORK_CONFIG.OUTPUT_SIZE})`,
  );
  console.log(
    `Training samples: ${Object.keys(Neural_Network_Train_Data).length}`,
  );
}

let dZ3 = []; // Output layer: ∂L/∂z³
let dW3 = []; // Output layer: ∂L/∂W³
let dB3 = []; // Output layer: ∂L/∂b³
let dZ2 = null; // Hidden layer 2: ∂L/∂z²
let dW2 = []; // Hidden layer 2: ∂L/∂W²
let dB2 = []; // Hidden layer 2: ∂L/∂b²
let dZ1 = null; // Hidden layer 1: ∂L/∂z¹
let dW1 = []; // Hidden layer 1: ∂L/∂W¹
let dB1 = []; // Hidden layer 1: ∂L/∂b¹

let m = Object.keys(Neural_Network_Train_Data).length;

let graphdata = ``;

// Main training function
function train_neural_network() {
  dZ3 = []; // Output layer: ∂L/∂z³
  dW3 = []; // Output layer: ∂L/∂W³
  dB3 = []; // Output layer: ∂L/∂b³
  dZ2 = null; // Hidden layer 2: ∂L/∂z²
  dW2 = []; // Hidden layer 2: ∂L/∂W²
  dB2 = []; // Hidden layer 2: ∂L/∂b²
  dZ1 = null; // Hidden layer 1: ∂L/∂z¹
  dW1 = []; // Hidden layer 1: ∂L/∂W¹
  dB1 = []; // Hidden layer 1: ∂L/∂b¹

  function back_propogation() {
    let xA_0; // Current input (pixel data)
    let xA_1; // Current hidden layer 1 activation
    let xA_2; // Current hidden layer 2 activation
    let xA_3; // Current output layer activation (before softmax)

    let total_xA_0 = []; // All input activations
    let total_xA_1 = []; // All hidden layer 1 activations
    let total_xA_2 = []; // All hidden layer 2 activations

    let softmax_xA_3; // Softmax output for gradient calculation
    let cost_data = []; // Cross-entropy loss for each training example
    let accuracy = 0; // Count of correct predictions

    for (const key in Neural_Network_Train_Data) {
      xA_0 = Neural_Network_Train_Data[key][0];
      total_xA_0.push(xA_0);

      // Forward propagation: Input -> Hidden Layer 1
      xA_1 = forward_propogation(
        Neural_Network_Train_Data[key][0],
        W_1,
        B_1,
        "leaky_relu",
      );
      total_xA_1.push(xA_1);

      // Forward propagation: Hidden Layer 1 -> Hidden Layer 2
      xA_2 = forward_propogation(xA_1, W_2, B_2, "leaky_relu");
      total_xA_2.push(xA_2);

      // Forward propagation: Hidden Layer 2 -> Output Layer
      xA_3 = forward_propogation(xA_2, W_3, B_3, "");

      // Apply softmax to get probability distribution
      let logits = xA_3; // output of linear layer

      // softmax probabilities
      let probs = softmax(logits);

      let label = Neural_Network_Train_Data[key][1][0];

      cost_data.push(Math.log(probs[label] + EPSILON));

      let pred = probs.indexOf(Math.max(...probs));
      if (pred === label) accuracy++;

      let dZ3_i = probs.slice();
      dZ3_i[label] -= 1;
      dZ3.push(dZ3_i);
    }

    // Cost = -(1/m) * Σ log(p_correct)
    let new_cost = 0;
    for (let gh = 0; gh < cost_data.length; gh++) {
      new_cost = new_cost + cost_data[gh];
    }
    new_cost = -(new_cost / m);

    console.log("The cost is " + new_cost);
    console.log("The accuracy is " + (accuracy / m) * 100);

    if (Math.random() < 0.1) {
      // 10% of iterations
      let weak_neurons_1 = total_xA_1[0].filter(
        (a) => Math.abs(a) < 0.01,
      ).length;
      let weak_neurons_2 = total_xA_2[0].filter(
        (a) => Math.abs(a) < 0.01,
      ).length;
      console.log(
        `Weak neurons (<0.01) - Layer1: ${weak_neurons_1}/${A_1_length}, Layer2: ${weak_neurons_2}/${A_2_length}`,
      );
    }

    cost_value_box.innerHTML =
      "The Cost Is " +
      new_cost +
      "<br>" +
      "The Accuracy Is " +
      (accuracy / m) * 100;

    graphdata +=
      `the cost is ` +
      new_cost +
      `\n` +
      `The accuracy is ` +
      (accuracy / m) * 100 +
      `\n`;
    const lines = graphdata.split("\n");
    if (lines.length > 2000) {
      graphdata = lines.slice(-2000).join("\n");
    }
    graphgenerater();

    // dW3 = (1/m) * dZ3^T * A_2
    let multiply_dZ3_xA_2T;
    multiply_dZ3_xA_2T = matrix_multipilcation_with_transpose(
      dZ3,
      transposeMatrix(total_xA_2),
    );

    for (let r = 0; r < multiply_dZ3_xA_2T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ3_xA_2T[r].length; l++) {
        divide_by_r.push(multiply_dZ3_xA_2T[r][l] / m);
      }
      dW3.push(divide_by_r);
    }

    // dB3 = (1/m) * Σ dZ3
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

    // dZ2 = (W3^T * dZ3) ⊙ g'(Z2)
    let matrix_of_W_3 = [];
    let g_sum = [];
    for (let g = 0; g < W_3.length; g++) {
      if (g_sum.length === A_2_length) {
        matrix_of_W_3.push(g_sum);
        g_sum = [];
      }
      g_sum.push(W_3[g]);
    }

    if (g_sum.length > 0) {
      matrix_of_W_3.push(g_sum);
    }
    matrix_of_W_3 = transposeMatrix(matrix_of_W_3);

    let multiply_of_W3T_dZ3 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_3),
      dZ3,
    );

    let A_2_derivative = derivative_leaky_relu(total_xA_2);

    dZ2 = element_wise_multiplication(multiply_of_W3T_dZ3, A_2_derivative);

    // dW2 = (1/m) * dZ2^T * A_1
    let multiply_dZ2_xA_2T;
    multiply_dZ2_xA_2T = matrix_multipilcation_with_transpose(
      dZ2,
      transposeMatrix(total_xA_1),
    );
    for (let r = 0; r < multiply_dZ2_xA_2T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ2_xA_2T[r].length; l++) {
        divide_by_r.push(multiply_dZ2_xA_2T[r][l] / m);
      }
      dW2.push(divide_by_r);
    }

    // dB2 = (1/m) * Σ dZ2
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

    // dZ1 = (W2^T * dZ2) ⊙ g'(Z1)
    let matrix_of_W_2 = [];
    let q_sum = [];
    for (let g = 0; g < W_2.length; g++) {
      if (q_sum.length === A_1_length) {
        matrix_of_W_2.push(q_sum);
        q_sum = [];
      }
      q_sum.push(W_2[g]);
    }
    if (q_sum.length > 0) {
      matrix_of_W_2.push(q_sum);
    }
    matrix_of_W_2 = transposeMatrix(matrix_of_W_2);

    let multiply_of_W2T_dZ2 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_2),
      dZ2,
    );

    let A_1_derivative = derivative_leaky_relu(total_xA_1);

    dZ1 = element_wise_multiplication(multiply_of_W2T_dZ2, A_1_derivative);

    // dW1 = (1/m) * dZ1^T * A_0
    let multiply_dZ1_xA_1T;
    multiply_dZ1_xA_1T = matrix_multipilcation_with_transpose(
      dZ1,
      transposeMatrix(total_xA_0),
    );
    for (let r = 0; r < multiply_dZ1_xA_1T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ1_xA_1T[r].length; l++) {
        divide_by_r.push(multiply_dZ1_xA_1T[r][l] / m);
      }
      dW1.push(divide_by_r);
    }

    // dB1 = (1/m) * Σ dZ1
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
  }

  back_propogation();

  // New parameter = Old parameter - (learning_rate * gradient)
  function update_parameters() {
    W_1 = W_update(W_1, alpha, dW1); // Update layer 1 weights
    W_2 = W_update(W_2, alpha, dW2); // Update layer 2 weights
    W_3 = W_update(W_3, alpha, dW3); // Update layer 3 weights
    B_1 = B_update(B_1, alpha, dB1); // Update layer 1 biases
    B_2 = B_update(B_2, alpha, dB2); // Update layer 2 biases
    B_3 = B_update(B_3, alpha, dB3); // Update layer 3 biases
  }

  update_parameters();
}

let newinterval = null;

function neural_network_main() {
  if (newinterval) {
    clearInterval(newinterval);
    newinterval = null;
  }

  button.forEach((btn) => {
    const intensity = parseFloat(btn.dataset.intensity) || 0;
    A_0.push(intensity);
  });

  A_1 = forward_propogation(A_0, W_1, B_1, "leaky_relu");
  A_2 = forward_propogation(A_1, W_2, B_2, "leaky_relu");
  A_3 = forward_propogation(A_2, W_3, B_3, "");

  const softmax_output = softmax(A_3);
  console.log(softmax_output);

  const maxProb = Math.max(...softmax_output);
  const predictedDigit = softmax_output.indexOf(maxProb);
  output_text.innerHTML = `<strong>Predicted: ${predictedDigit}</strong>`;

  const percentageOutput = convertToPercentages(softmax_output);
  for (const item of percentageOutput) {
    const percent = item.percentage.toFixed(2);
    output_text.innerHTML += `<br>${item.index} = ${percent}%`;
  }

  A_0 = [];
  A_1 = [];
  A_2 = [];
  A_3 = [];
}

function noisefunction(image, width, height, noiseLevel = 0.005) {
  return image.map((pixel) => {
    const u1 = Math.random();
    const u2 = Math.random();
    const gaussianNoise =
      Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const noisyPixel = pixel + gaussianNoise * noiseLevel;
    return Math.max(0, Math.min(1, noisyPixel));
  });
}

function position(image, height, width, dx, dy) {
  const grid = [];
  for (let row = 0; row < height; row++) {
    grid.push(image.slice(row * width, (row + 1) * width));
  }

  const translated = [];

  for (let y = 0; y < height; y++) {
    translated[y] = [];
    for (let x = 0; x < width; x++) {
      const srcX = x - dx;
      const srcY = y - dy;

      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        translated[y][x] = grid[srcY][srcX];
      } else {
        translated[y][x] = 0;
      }
    }
  }

  return translated.flat();
}

function rotation(image, angleDegrees, height, width) {
  const angleRad = (angleDegrees * Math.PI) / 180;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  const centerX = width / 2;
  const centerY = height / 2;

  const rotated = new Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const relX = x - centerX;
      const relY = y - centerY;
      const srcX = Math.round(relX * cos - relY * sin + centerX);
      const srcY = Math.round(relX * sin + relY * cos + centerY);
      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        rotated[y * width + x] = image[srcY * width + srcX];
      } else {
        rotated[y * width + x] = 0; // Background
      }
    }
  }

  return rotated;
}

function randomnessadder(sourceData) {
  const gridSize = NETWORK_CONFIG.GRID_WIDTH;

  const augmentedDataset = JSON.parse(JSON.stringify(sourceData));

  for (const key in sourceData) {
    let augmented = sourceData[key][0].slice(); // Copy array

    const dx = randomrangenumber(-8, 8);
    const dy = randomrangenumber(-2, 2);
    augmented = position(augmented, gridSize, gridSize, dx, dy);

    augmented = noisefunction(augmented, gridSize, gridSize, 0.03);

    const angle = randomrangenumber(-5, 5);
    augmented = rotation(augmented, angle, gridSize, gridSize);

    augmented = augmented.map((val) => (val == null || isNaN(val) ? 0 : val));

    augmentedDataset[key][0] = augmented;
  }

  return augmentedDataset;
}

function clear_drawing() {
  button.forEach((btn) => {
    btn.dataset.intensity = 0;
    btn.style.background = "rgb(255, 255, 255)";
  });
  output_text.innerHTML =
    '<p class="placeholder-text">Draw a digit and click "Recognize"</p>';
}

let neuralnetworkdatacopy = null;

if (typeof Neural_Network_Train_Data !== "undefined") {
  neuralnetworkdatacopy = JSON.parse(JSON.stringify(Neural_Network_Train_Data));
}

function add_own_drawing_to_data() {
  const labelInput = document.getElementById("custom_label_input");
  const labelVal = parseInt(labelInput.value, 10);

  if (isNaN(labelVal) || labelVal < 0 || labelVal > 9) {
    alert("Please enter a valid digit (0-9) for this drawing.");
    return;
  }

  let new_A_0 = [];
  button.forEach((btn) => {
    const intensity = parseFloat(btn.dataset.intensity) || 0;
    new_A_0.push(intensity);
  });

  const newKey = "custom_" + Date.now();

  Neural_Network_Train_Data[newKey] = [new_A_0, [labelVal]];

  if (neuralnetworkdatacopy) {
    neuralnetworkdatacopy[newKey] = [new_A_0, [labelVal]];
  }

  m = Object.keys(Neural_Network_Train_Data).length;

  alert(`Added digit ${labelVal} to the training data! Total samples: ${m}`);
  clear_drawing();
  labelInput.value = "";
}

window._previewState = { active: false, original: null };

function previewAugmentation() {
  const boxes = document.querySelectorAll(".box");
  if (!boxes || !boxes.length) return;

  const btn = document.getElementById("previewAugBtn");

  if (!window._previewState.active) {
    const original = Array.from(boxes).map(
      (b) => parseFloat(b.dataset.intensity) || 0,
    );
    window._previewState.original = original;

    let augmented = null;
    try {
      if (
        typeof Neural_Network_Train_Data !== "undefined" &&
        Object.keys(Neural_Network_Train_Data).length
      ) {
        const keys = Object.keys(Neural_Network_Train_Data);
        const randKey = keys[Math.floor(Math.random() * keys.length)];
        const sample = Neural_Network_Train_Data[randKey];
        augmented = sample && sample[0] ? sample[0].slice() : null;
      }
    } catch (e) {
      augmented = null;
    }

    if (!augmented) {
      const src = { preview_sample: [original.slice(), [0]] };
      const augmentedObj = randomnessadder(src);
      augmented = augmentedObj.preview_sample[0];
    }

    boxes.forEach((b, i) => {
      const v = augmented[i] || 0;
      b.dataset.intensity = v;
      const gray = Math.round((1 - v) * 255);
      b.style.background = `rgb(${gray}, ${gray}, ${gray})`;
    });

    window._previewState.active = true;
    if (btn) btn.innerHTML = '<i class="fas fa-undo"></i> Restore Original';
  } else {
    const orig =
      window._previewState.original || Array.from(boxes).map(() => 0);
    boxes.forEach((b, i) => {
      const v = orig[i] || 0;
      b.dataset.intensity = v;
      const gray = Math.round((1 - v) * 255);
      b.style.background = `rgb(${gray}, ${gray}, ${gray})`;
    });

    window._previewState.active = false;
    window._previewState.original = null;
    if (btn) btn.innerHTML = '<i class="fas fa-eye"></i> Draw From Dataset';
  }
}

let training_length = 20; // Number of training iterations
let loadpercent = 0; // Progress bar percentage
let useDataAugmentation = true; // Enable/disable data augmentation
let train_interval = null; // Reference to training interval

loadbar.style.width = `calc(90% / ${training_length} * ${loadpercent})`;

let isTraining = false;

function train_button() {
  if (isTraining) {
    console.log("Training already in progress...");
    return;
  }

  isTraining = true;
  loadpercent = 0;

  loadbarcontain.style.display = "flex";
  loadbar.style.width = "0%";

  function trainIteration(epoch) {
    if (epoch >= training_length) {
      loadbarcontain.style.display = "none";
      loadpercent = 0;
      isTraining = false;
      console.log("Training complete!");
      return;
    }

    if (useDataAugmentation) {
      Neural_Network_Train_Data = randomnessadder(neuralnetworkdatacopy);
    }

    train_neural_network();

    dZ3 = [];
    dW3 = [];
    dB3 = [];
    dZ2 = null;
    dW2 = [];
    dB2 = [];
    dZ1 = null;
    dW1 = [];
    dB1 = [];

    loadpercent = epoch + 1;
    const progressPercent = (loadpercent / training_length) * 90;
    loadbar.style.width = `${progressPercent}%`;

    setTimeout(() => trainIteration(epoch + 1), 0);
  }

  trainIteration(0);
}

alpha_submit.addEventListener("click", () => {
  let newAlpha = parseFloat(alpha_value.value) || 0.1;
  let newLambda =
    parseFloat(document.getElementById("lambda_value").value) || 0.000001;
  let newIterations = parseInt(train_length_input.value, 10) || 20;

  if (newAlpha <= 0 || newAlpha > 10) {
    alert("Learning rate must be between 0.0001 and 10");
    newAlpha = 0.1;
  }

  if (newLambda < 0 || newLambda > 1) {
    alert("Lambda must be between 0 and 1");
    newLambda = 0.000001;
  }

  if (newIterations < 1 || newIterations > 10000) {
    alert("Training iterations must be between 1 and 10000");
    newIterations = 20;
  }

  alpha = newAlpha;
  lambda = newLambda;
  training_length = newIterations;

  console.log(
    `Hyperparameters updated: α=${alpha}, λ=${lambda}, iterations=${training_length}`,
  );
});

const augmentationToggle = document.getElementById("augmentation_toggle");
const augmentationStatus = document.getElementById("augmentation_status");

if (augmentationToggle) {
  augmentationToggle.addEventListener("change", () => {
    useDataAugmentation = augmentationToggle.checked;
    if (augmentationStatus) {
      augmentationStatus.textContent = useDataAugmentation
        ? "Enabled"
        : "Disabled";
      augmentationStatus.classList.toggle("active", useDataAugmentation);
    }
    console.log(
      "Data Augmentation: " + (useDataAugmentation ? "Enabled" : "Disabled"),
    );
  });
}
