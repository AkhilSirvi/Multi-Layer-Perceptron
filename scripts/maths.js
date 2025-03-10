function TanH(z) {
  let tanhvalue = (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
  if (isNaN(tanhvalue) && z > 0) {
    return 1;
  } else if (isNaN(tanhvalue) && z < 0) {
    return -1;
  } else {
    return tanhvalue;
  }
}

function log(z) {
  return Math.log(z);
}
// the parameter in the random function decimal that how many decimal point we want of random number
function random(decimals) {
  var randomNumber = Math.random() * 2 - 1;
  return parseFloat(randomNumber.toFixed(decimals));
}

function forward_propogation(M1, M2, B, Activation_function) {
  // M1 and M2 are going to be in array
  let multiplication_data = 0;
  let data_array = [];
  let shape_first_matrix_row = M1.length;
  let k = 0;
  let v = 0;
  for (let i = 0; i < M2.length; i++) {
    multiplication_data = multiplication_data + M1[k] * M2[i];

    k++;
    if (k == shape_first_matrix_row) {
      k = 0;
      if (Activation_function == "TanH") {
        data_array.push(TanH(multiplication_data + B[v]));
        v++;
      }
      multiplication_data = 0;
    }
  }
  return data_array;
}

function softmax(z) {
  let data = [];
  let expsum = 0;
  for (let k = 0; k < z.length; k++) {
    expsum += Math.exp(z[k]);
  }
  for (let i = 0; i < z.length; i++) {
    data.push(Math.exp(z[i]) / expsum);
  }
  return data;
}

function transposeMatrix(matrix) {
  const transposedMatrix = matrix[0].map((_, colIndex) =>
    matrix.map((row) => row[colIndex])
  );

  return transposedMatrix;
}
// this will multiply the matrix with row to column
function matrix_multipilcation_with_transpose(M1, M2) {
  let output = [];
  let rows1 = M1.length;
  let rows2 = M2.length;
  let colum1 = M1[0].length;
  let colum2 = M2[0].length;
  if (rows1 !== colum2) {
    console.log(
      "Error: The number of columns in Matrix 1 must match the number of rows in Matrix 2."
    );
    return output;
  }
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

function derivative_tanH(h) {
  let square_matrix = h.map((innerArr) => innerArr.map((value) => value ** 2));
  let newArray = square_matrix.map((subArray) =>
    subArray.map((value) => 1 - value)
  );
  return newArray;
}

function element_wise_multiplication(J1, J2) {
  const result = J1.map((row, i) => row.map((num, j) => num * J2[i][j]));
  return result;
}

let lambda = 0.000001

function W_update(intialmatrix, learning_rate, backpropvalue) {
  let flat_backpropvalue = transposeMatrix(backpropvalue).flat();
  let multipliedMatrix = flat_backpropvalue.map(function (value) {
    return value * learning_rate;
  });
  let result = intialmatrix.map(
    (value, index) => value - multipliedMatrix[index]
  );

  let regulazationmatrix = intialmatrix.map(function (value) {
    return value * lambda
  });

  let regulazationanswer = result.map((value, index) => value + regulazationmatrix[index]);

  return regulazationanswer;
}

function B_update(intialmatrix, learning_rate, backpropvalue) {
  let multipliedMatrix = backpropvalue.map(function (value) {
    return value * learning_rate;
  });
  let result = intialmatrix.map(
    (value, index) => value - multipliedMatrix[index]
  );
  return result;
}

function convertToPercentages(arr) {
  // Calculate the percent
  const sum = arr.reduce((total, num) => total + num, 0);
  const percentages = arr.map((num, index) => ({
    index: index,
    percentage: (num / sum) * 100,
  }));
  // Sort the percentages in descending order based on the percentage value
  percentages.sort((a, b) => b.percentage - a.percentage);
  return percentages;
}
