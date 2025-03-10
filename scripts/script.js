let second_box = document.getElementById("second");
let output_text = document.getElementById("output_text");
let loadbar = document.getElementById("loadbarmain");
let loadbarcontain = document.getElementById("loadbar");
let cost_value_box = document.getElementById("cost_value_output");
let alpha_value = document.getElementById("alpha_value");
let train_length_input = document.getElementById("train_length_input");
let alpha_submit = document.getElementById("alpha_submit");

// This Fuction will create element in html which we can draw something on
function pixel_creater() {
  for (let pixel_creater_i = 0; pixel_creater_i < 400; pixel_creater_i++) {
    let pixel = document.createElement("div");
    second_box.appendChild(pixel);
    let pixel_creater_v = 1;
    let pixel_creater_k = 1;
    pixel_creater_v++;
    if (pixel_creater_v == 10) {
      pixel_creater_k++;
      pixel_creater_v = 0;
    }
    pixel.id = pixel_creater_v + "rowbox" + pixel_creater_k;
    pixel.className = "box";
    pixel.draggable = false;
  }
}
pixel_creater();

let body = document.body;
let button = document.querySelectorAll(".box");
let click = false;

// these two event listener will check if you pressed left click of mouse
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
// this event will turn the button to black on left click with mouse hovered on
button.forEach((button) => {
  button.addEventListener("mouseover", () => {
    if (click == true) {
      button.style.background = "black";
    }
  });

  second_box.addEventListener("touchmove", (event) => {
   event.preventDefault();
    const touch = event.touches[0];
    const x = touch.clientX;
    const y = touch.clientY;
    const rect = button.getBoundingClientRect();

    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      button.style.background = "black";
    }

  });
});

// these all variable are to declare the layers and their weight and bais
let A_0_length = 400;
let A_1_length = 16 ;
let A_2_length = 16;
let A_3_length = 10;

let A_0 = [];
let A_1 = [];
let A_2 = [];
let A_3 = [];



// alpha is the learning rate
let alpha = 0.1;

// these all function will initialise the random value for the weight and bais at start
function W_1_function_random_no() {
  let W_1_length = A_1_length * A_0_length;
  for (let i = 0; i < W_1_length; i++) {
    W_1.push(random(2));
  }
}
function B_1_function_random_no() {
  let B_1_length = A_1_length;
  for (let i = 0; i < B_1_length; i++) {
    B_1.push(random(2));
  }
}
function W_2_function_random_no() {
  let W_2_length = A_1_length * A_2_length;
  for (let i = 0; i < W_2_length; i++) {
    W_2.push(random(2));
  }
}
function B_2_function_random_no() {
  let B_2_length = A_2_length;
  for (let i = 0; i < B_2_length; i++) {
    B_2.push(random(2));
  }
}

function W_3_function_random_no() {
  let W_3_length = A_2_length * A_3_length;
  for (let i = 0; i < W_3_length; i++) {
    W_3.push(random(2));
  }
}
function B_3_function_random_no() {
  let B_3_length = A_3_length;
  for (let i = 0; i < B_3_length; i++) {
    B_3.push(random(2));
  }
}

// we are not initialising the weight and bias randomly because we alredy have train data in data js stored
// W_1_function_random_no();
// B_1_function_random_no();
// W_2_function_random_no();
// B_2_function_random_no();
// W_3_function_random_no();
// B_3_function_random_no();

// these are the variables for the back prop.
let dZ3 = [];
let dW3 = [];
let dB3 = [];
let dZ2 = null;
let dW2 = [];
let dB2 = [];
let dZ1 = null;
let dW1 = [];
let dB1 = [];
let m = Object.keys(Neural_Network_Train_Data).length;

let graphdata = ``;

// This function "train_neural_network" will run two function. One is back prop. and other one is update parameters
function train_neural_network() {
  function back_propogation() {
    let xA_0;
    let xA_1;
    let xA_2;
    let xA_3;
    let total_xA_0 = [];
    let total_xA_1 = [];
    let total_xA_2 = [];
    let softmax_xA_3;
    let cost_data = [];
    let accuracy = 0;
    // this is used for dZ3 calculation
    for (const key in Neural_Network_Train_Data) {
      xA_0 = Neural_Network_Train_Data[key][0];
      total_xA_0.push(xA_0);
      xA_1 = forward_propogation(
        Neural_Network_Train_Data[key][0],
        W_1,
        B_1,
        "TanH"
      );
      total_xA_1.push(xA_1);
      xA_2 = forward_propogation(xA_1, W_2, B_2, "TanH");
      total_xA_2.push(xA_2);
      xA_3 = forward_propogation(xA_2, W_3, B_3, "TanH");
      softmax_xA_3 = softmax(xA_3);
      othersoftmax = softmax(xA_3);
      softmax_xA_3[Neural_Network_Train_Data[key][1]] =
        softmax_xA_3[Neural_Network_Train_Data[key][1]] - 1;
      dZ3.push(softmax_xA_3);

      cost_data.push(log(othersoftmax[Neural_Network_Train_Data[key][1]] * 1));
      
      if (othersoftmax[Neural_Network_Train_Data[key][1]] == Math.max.apply(null, othersoftmax)) {
        accuracy++
      }
    }

    let new_cost = 0;
    for (let gh = 0; gh < cost_data.length; gh++) {
      new_cost = new_cost + cost_data[gh];
    }

    new_cost = -(new_cost / m);

    console.log("The cost is " + new_cost);
    console.log("The accuracy is " + (accuracy/m)*100)
    cost_value_box.innerHTML = "The Cost Is " + new_cost + "<br>" + "The Accuracy Is " + (accuracy/m)*100;
    graphdata += `the cost is ` + new_cost + `\n` + `The accuracy is ` + (accuracy/m)*100 + `\n`;
    graphgenerater();

    // this is used of dW3 calculation
    let multiply_dZ3_xA_2T;
    multiply_dZ3_xA_2T = matrix_multipilcation_with_transpose(
      dZ3,
      transposeMatrix(total_xA_2)
    );

    for (let r = 0; r < multiply_dZ3_xA_2T.length; r++) {
      let divide_by_r = [];
      for (let l = 0; l < multiply_dZ3_xA_2T[r].length; l++) {
        divide_by_r.push(multiply_dZ3_xA_2T[r][l] / m);
      }
      dW3.push(divide_by_r);
    }

    // this is used of dB3 calculation
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
    // this is used for dZ2 calculation
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
    let multiply_of_W3T_dZ3 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_3),
      dZ3
    );
    let A_2_derivative = derivative_tanH(total_xA_2);
    dZ2 = element_wise_multiplication(multiply_of_W3T_dZ3, A_2_derivative);

    // this is used for dW2 calculation
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

    // this is used for dB2 calculation
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
    // this is used for dZ1 calculation
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
    let multiply_of_W2T_dZ2 = matrix_multipilcation_with_transpose(
      transposeMatrix(matrix_of_W_2),
      dZ2
    );
    let A_1_derivative = derivative_tanH(total_xA_1);
    dZ1 = element_wise_multiplication(multiply_of_W2T_dZ2, A_1_derivative);

    // this is used for dW1 calculation
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

    // this is used for dB1 calculation
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

    //last end of back prop
  }

  back_propogation();

  function update_parameters() {
    W_1 = W_update(W_1, alpha, dW1);
    W_2 = W_update(W_2, alpha, dW2);
    W_3 = W_update(W_3, alpha, dW3);
    B_1 = B_update(B_1, alpha, dB1);
    B_2 = B_update(B_2, alpha, dB2);
    B_3 = B_update(B_3, alpha, dB3);
  }
  update_parameters();
}

function neural_network_main() {
  newinterval = setInterval(() => {
    button.forEach((button) => {
      if (button.style.background != "black") {
        A_0.push(0);
      } else {
        A_0.push(1);
      }
    });

    A_1 = forward_propogation(A_0, W_1, B_1, "TanH");
    A_2 = forward_propogation(A_1, W_2, B_2, "TanH");
    A_3 = forward_propogation(A_2, W_3, B_3, "TanH");

    // console.log(W_1);
    // console.log(A_1);
    // console.log(A_2);
    // console.log(A_3);
    console.log(softmax(A_3));

    // this will print the output from the highest value from softmax of the last layer
    let softmax_output = softmax(A_3);
    for (let j = 0; j < softmax_output.length; j++) {
      if (softmax_output[j] == Math.max.apply(null, softmax_output)) {
        output_text.innerHTML = "The output is " + j;
      }
    }

    // this will print the other possible outputs with their percentage of what the model is going to give answer
    let percentageoutput = convertToPercentages(softmax(A_3));
    let percentageoutput_index = percentageoutput.map((item) => item.index);
    let percentageoutput_percent = percentageoutput.map(
      (item) => item.percentage
    );
    for (let u = 0; u < percentageoutput.length; u++) {
      output_text.innerHTML +=
        "<br>" + percentageoutput_index[u] + "=" + percentageoutput_percent[u];
    }

    A_0 = [];
    A_1 = [];
    A_2 = [];
    A_3 = [];
  }, 100);
  // this is for - if we want to reset the drawing of the numbers
  button.forEach((button) => {
    button.style.background = `white`;
  });
}



function noisefunction(inputarray, width, height, biasFactor) {
  let noisedarray = [];

  for (let noise_loop = 0; noise_loop < width*height; noise_loop++) {
    if (inputarray[noise_loop] === 0) {     
      noisedarray.push( Math.random() < biasFactor ? 0 : 1)
    }
    else{
      noisedarray.push(1)
    }
  }

  return noisedarray;
}

function position(inputarray, height, width, dxv, dyv) {
  const originalArray = Array.from({ length: height }, (_, i) => inputarray.slice(i * width, (i + 1) * width));
  const dx = dxv;
  const dy = dyv;
  
  const rows = originalArray.length;
  const columns = originalArray[0].length;

  const translatedArray = [];

  for (let i = 0; i < rows; i++) {
    translatedArray[i] = [];
    for (let j = 0; j < columns; j++) {
      const originalX = j;
      const originalY = i;
      const translatedX = originalX + dx;
      const translatedY = originalY + dy;
      if (translatedX >= 0 && translatedX < columns && translatedY >= 0 && translatedY < rows) {
        translatedArray[i][j] = originalArray[translatedY][translatedX];
      } 
      else {
        translatedArray[i][j] = 0;
      }
    }
  }
  return translatedArray.flat();
}

function rotation(inputarray, anglevalue, height, width) {
  let angle = (anglevalue * Math.PI) / 180
  
  const centerX = width / 2;
  const centerY = height / 2;

  const rotatedarray = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const relX = x - centerX;
      const relY = y - centerY;
        
      // Apply the 2d rotation matrix
      const newX = Math.round(relX * Math.cos(angle) - relY * Math.sin(angle));
      const newY = Math.round(relX * Math.sin(angle) + relY * Math.cos(angle));

      const absX = Math.floor(newX + centerX);
      const absY = Math.floor(newY + centerY);
  
      // Get the pixel value from the original code
      const pixel = inputarray[absY * width + absX];
  
      // Set the corresponding pixel value in the rotated code
      rotatedarray[y * width + x] = pixel;
    }
  }
  
  return rotatedarray;
}
function randomrangenumber(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min
}
function randomnessadder(originaldata, copydata) {
  for (const key in copydata) {
    let noisedarray1 = position(copydata[key][0], 20, 20, randomrangenumber(-4, 4), randomrangenumber(-4, 4))
    let noisedarray2 = noisefunction(noisedarray1, 20, 20, 0.99)
    let noisedarray3 = rotation(noisedarray2, randomrangenumber(-15, 15), 20, 20)
    let noisedarray4 = noisedarray3.map((value) => {
      if (value === null || value === undefined) {
        return 0;
      } else {
        return value;
      }
    })
    originaldata[key][0] = noisedarray4;
  }
}

const neuralnetworkdatacopy = JSON.parse(JSON.stringify(Neural_Network_Train_Data))






let training_length = 20;
let loadpercent = 0;
loadbar.style.width = "calc(90%/" + training_length + "*" + loadpercent + ")";

let train_interval = null;

// this is for that it can display the loading bar and stop it when it reach the length of the variable
function train_button() {
  loadbarcontain.style.display = "flex";
  for (let trainingloop = 0; trainingloop < training_length; trainingloop++) {

    randomnessadder(Neural_Network_Train_Data, neuralnetworkdatacopy);
    train_neural_network()
    
    dZ3 = [];
    dW3 = [];
    dB3 = [];
    dZ2 = null;
    dW2 = [];
    dB2 = [];
    dZ1 = null;
    dW1 = [];
    dB1 = [];
    
    loadpercent++;
    loadbar.style.width = "calc(90%/" + training_length + "*" + loadpercent + ")";
    if (training_length == loadpercent) {
      loadbarcontain.style.display = "none";
      loadpercent = 0;
    }
  }
}





alpha_submit.addEventListener("click", () => {
  alpha = parseFloat(alpha_value.value);
  lambda = parseFloat(document.getElementById("lambda_value").value)
  training_length = parseFloat(train_length_input.value);
});