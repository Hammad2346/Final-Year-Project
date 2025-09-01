let network = [];
let input = [];
let layernumber = 0;
let positions = [];
let images = [];

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b','c', 'd', 'e', 
                       'f', 'g', 'h','i','j','k','l','m', 'n','o','p', 'q', 
                       'r', 's', 't','u','v','w','x','y','z']

function preload() {
  modelPromise = fetch("emnist_model.json").then(res => res.json());
  csvPromise = fetch("emnist_data.csv").then(res => res.text());
}

async function setup() {
  createCanvas(1100, 600);
  noLoop();
  
  const model = await modelPromise;
  const csvText = await csvPromise;
  images = csvText.trim().split('\n').slice(1);
  
  input = getRandomInput(); 
  network = buildNetwork(model);
  
  getPositions();
  drawNeurons();
  
  
   
}


document.addEventListener("keydown", function (event) {
  if (layernumber < network.length) {
    if(layernumber===0){
      background(30)

      drawLayer(0);
      layernumber = 1;
    }
    const activations = getActivationValuesOfLayer(layernumber, input);
    drawLayer( layernumber);
    layernumber++;
    input = activations;
    console.log(input)
  }
});

class Neuron {
  constructor(weights, bias) {
    this.weights = weights;
    this.bias = bias;
    this.alpha = 0.01;
    this.neuronValue = null;
    this.x = 0;
    this.y = 0;
  }

  calculate(input) {
    const z = this.dot(input, this.weights) + this.bias;
    this.neuronValue = this.activation(z);
    return this.neuronValue
  }
  
  activation(z) {
    return z < 0 ? this.alpha * z : z;
  }
  
  dot(a, b) {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  }
}

class InputNeuron {
  constructor(value = 0) {
    this.neuronValue = value;
    this.x = 0;
    this.y = 0;
  }
}

function buildNetwork(model) {
  const hls_weights = model.hidden_layer_weights;
  const hls_biases = model.hidden_layer_biases;
  const ol_weights = model.output_layer_weights;
  const ol_biases = model.output_layer_biases;
  
  const network = [];
  
  const inputLayer = [];
  for (let i = 0; i < input.length; i++) {
    inputLayer.push(new InputNeuron(input[i]));
  }
  network.push(inputLayer);

  for (let i = 0; i < hls_weights.length; i++) {
    const layer = [];
    for (let j = 0; j < hls_weights[i].length; j++) {
      layer.push(new Neuron(hls_weights[i][j], hls_biases[i][j]));
    }
    network.push(layer);
  }
  
  const outputLayer = [];
  for (let i = 0; i < ol_weights.length; i++) {
    outputLayer.push(new Neuron(ol_weights[i], ol_biases[i]));
  }
  network.push(outputLayer);
  
  return network;
}

function getRandomInput() {
  const randomImg = images[Math.floor(Math.random() * images.length)];
  const parts = randomImg.split(',').map(Number);
  return parts.slice(1).map(p => p / 255); 
}

function getActivationValuesOfLayer(i, input) {
  const layer = network[i];
  return layer.map(neuron => neuron.calculate(input));
}

function getPositions() {
  const layerSpacing = width / (network.length + 1);
  const heightInPixels = height * 0.8;
  
  for (let layerNum = 0; layerNum < network.length; layerNum++) {
    let x = layerSpacing * (layerNum + 1);
    let layer = network[layerNum];
    let neuronSpacing = heightInPixels / (layer.length + 1);
    for (let i = 0; i < layer.length; i++) {
      let y = height / 2 - heightInPixels / 2 + neuronSpacing * (i + 1);
      network[layerNum][i].x = x;
      network[layerNum][i].y = y;
    }
  }
}

function drawNeurons() {
  background(30);
  const top = 10;  
  const left = 200;                                 
  const neuronSize = 20;
  const gap = 5;

  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      const x = left + j * (neuronSize + gap);
      const y = top + i * (neuronSize + gap);
      const index = i * 28 + j;
      const opacity = input[index] * 255;

      stroke(255);
      fill(255, 255, 255, opacity);
      ellipse(x, y, neuronSize, neuronSize);
    }
  }
}
function indexOfMax(layer) {
    if (layer.length === 0) {
        return -1;
    }

    var max = layer[0].neuronValue;
    var maxIndex = 0;

    for (var i = 1; i < layer.length; i++) {
        if (layer[i].neuronValue > max) {
            maxIndex = i;
            max = layer[i].neuronValue;
        }
    }

    return maxIndex;
}

function drawLayer( layerIdx) {
  const threshold = 0.1; 
  
  for (let i = 0; i < network[layerIdx].length; i++) {
    if(layerIdx===0){
    const neuron = network[layerIdx][i];
    const activation = neuron.neuronValue;
    
    stroke(255); 
    
    if (activation > threshold) {
    
      const opacity =activation*100;
      fill(255, 255, 255, opacity);
    } else {
      
      noFill();
    }
    
    ellipse(neuron.x, neuron.y, 20, 20);}
    else{
      const neuron = network[layerIdx][i];
    const activation = neuron.neuronValue;
    const weights=neuron.weights;
    const prevLayer=network[layerIdx-1]
    
    stroke(255); 
    
    if (activation > threshold) {
    
      const opacity =activation*100;
      fill(255, 255, 255, opacity);
    } else {
      
      noFill();
    }
    if(layernumber===network.length-1&&i===indexOfMax(network[layerIdx])){
      textSize(15)
      fill(255)
      text(classes[int(indexOfMax(network[layerIdx]))],neuron.x+20,neuron.y+8)

      stroke(0,255,0)
      fill(255, 255, 255); //just put opacity there if want to make it look genuine

    }
    ellipse(neuron.x, neuron.y, 20, 20);

    for(let j=0;j<prevLayer.length;j++){
      const opacity=weights[i]*180
      if(weights[j]>0){
      stroke(0,255,0,opacity)
      }else{
        stroke(255,0,0,opacity)
      }

      line(neuron.x, neuron.y, prevLayer[j].x, prevLayer[j].y)
    }
    }
  }
  
}