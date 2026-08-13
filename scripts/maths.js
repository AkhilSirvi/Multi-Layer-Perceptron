"use strict";

const EPSILON = 1e-15;
const EXP_MAX = 709;

function TanH(z) {
  const expPos = Math.exp(z);
  const expNeg = Math.exp(-z);
  const result = (expPos - expNeg) / (expPos + expNeg);
  if (Number.isNaN(result)) {
    return z > 0 ? 1 : -1;
  }
  return result;
}

function relu(z) {
  return Math.max(0, z);
}

function leaky_relu(z) {
  return z > 0 ? z : 0.01 * z;
}

function log(z) {
  return Math.log(Math.max(z, EPSILON));
}

function random(decimals) {
  const value = Math.random() * 2 - 1;
  return Number(value.toFixed(decimals));
}

function randomrangenumber(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// forward propagation - does A[l] = σ(W[l] · A[l-1] + B[l])

function forward_propogation(inputs, weights, biases, activationFn) {
  const numInputs = inputs.length;
  const numOutputs = biases.length;
  const outputs = [];

  let weightIdx = 0;

  for (let neuron = 0; neuron < numOutputs; neuron++) {
    let weightedSum = 0;

    for (let input = 0; input < numInputs; input++) {
      weightedSum += inputs[input] * weights[weightIdx];
      weightIdx++;
    }

    const preActivation = weightedSum + biases[neuron];

    if (activationFn === "TanH") {
      outputs.push(TanH(preActivation));
    } else if (activationFn === "relu") {
      outputs.push(relu(preActivation));
    } else if (activationFn === "leaky_relu") {
      outputs.push(leaky_relu(preActivation));
    } else {
      outputs.push(preActivation);
    }
  }

  return outputs;
}

function softmax(logits) {
  const clampedLogits = logits.map((z) => Math.max(-100, Math.min(100, z)));
  const maxLogit = Math.max(...clampedLogits);
  const expValues = clampedLogits.map((z) => Math.exp(z - maxLogit));
  const expSum = expValues.reduce((sum, val) => sum + val, 0);
  return expValues.map((exp) => exp / (expSum + EPSILON));
}

function transposeMatrix(matrix) {
  if (!matrix || !matrix.length || !matrix[0]) {
    return [];
  }

  return matrix[0].map((_, colIdx) => matrix.map((row) => row[colIdx]));
}

function matrix_multipilcation_with_transpose(M1, M2) {
  const rows1 = M1.length;
  const cols1 = M1[0].length;
  const rows2 = M2.length;
  const cols2 = M2[0].length;

  if (rows1 !== cols2) {
    console.error(
      `Matrix dimension mismatch: M1 has ${rows1} rows but M2 has ${cols2} columns`,
    );
    return [];
  }

  const output = [];

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

function derivative_tanH(activations) {
  return activations.map((row) => row.map((a) => 1 - a * a));
}

function derivative_relu(activations) {
  return activations.map((row) => row.map((a) => (a > 0 ? 1 : 0)));
}

function derivative_leaky_relu(activations) {
  return activations.map((row) => row.map((a) => (a > 0 ? 1 : 0.01)));
}

function element_wise_multiplication(A, B) {
  return A.map((row, i) => row.map((val, j) => val * B[i][j]));
}

let lambda = 0.00001;

function W_update(weights, learningRate, gradients) {
  const flatGradients = transposeMatrix(gradients).flat();
  const scaledGradients = flatGradients.map((g) => g * learningRate);
  const updated = weights.map((w, i) => w - scaledGradients[i]);
  const regularization = weights.map((w) => w * (lambda * learningRate));
  return updated.map((w, i) => w - regularization[i]);
}

function B_update(biases, learningRate, gradients) {
  const scaledGradients = gradients.map((g) => g * learningRate);
  return biases.map((b, i) => b - scaledGradients[i]);
}

function convertToPercentages(scores) {
  const sum = scores.reduce((total, val) => total + val, 0);
  const percentages = scores.map((score, index) => ({
    index: index,
    percentage: (score / sum) * 100,
  }));
  return percentages.sort((a, b) => b.percentage - a.percentage);
}
