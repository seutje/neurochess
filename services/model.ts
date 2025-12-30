import * as tf from '@tensorflow/tfjs';
import { INPUT_PLANES, FILTERS, RESIDUAL_BLOCKS, POLICY_OUTPUT_SIZE, LEARNING_RATE } from '../constants';

// Helper for Residual Block
const residualBlock = (input: tf.SymbolicTensor): tf.SymbolicTensor => {
  let x = tf.layers.conv2d({
    filters: FILTERS,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }).apply(input) as tf.SymbolicTensor;
  
  x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
  x = tf.layers.reLU().apply(x) as tf.SymbolicTensor;
  
  x = tf.layers.conv2d({
    filters: FILTERS,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }).apply(x) as tf.SymbolicTensor;
  
  x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
  
  x = tf.layers.add().apply([x, input]) as tf.SymbolicTensor;
  x = tf.layers.reLU().apply(x) as tf.SymbolicTensor;
  
  return x;
};

export const createTinyZeroModel = (): tf.LayersModel => {
  const input = tf.input({ shape: [8, 8, INPUT_PLANES] });

  // 1. Convolutional Entry
  let x = tf.layers.conv2d({
    filters: FILTERS,
    kernelSize: 3,
    padding: 'same',
    useBias: false
  }).apply(input) as tf.SymbolicTensor;
  
  x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
  x = tf.layers.reLU().apply(x) as tf.SymbolicTensor;

  // 2. Residual Tower
  for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
    x = residualBlock(x);
  }

  // 3. Policy Head
  let p = tf.layers.conv2d({
    filters: 2,
    kernelSize: 1,
    padding: 'same',
    useBias: false
  }).apply(x) as tf.SymbolicTensor;
  
  p = tf.layers.batchNormalization().apply(p) as tf.SymbolicTensor;
  p = tf.layers.reLU().apply(p) as tf.SymbolicTensor;
  p = tf.layers.flatten().apply(p) as tf.SymbolicTensor;
  // Softmax over all possible moves
  const policyOutput = tf.layers.dense({
    units: POLICY_OUTPUT_SIZE,
    activation: 'softmax',
    name: 'policy_head'
  }).apply(p) as tf.SymbolicTensor;

  // 4. Value Head
  let v = tf.layers.conv2d({
    filters: 1,
    kernelSize: 1,
    padding: 'same',
    useBias: false
  }).apply(x) as tf.SymbolicTensor;
  
  v = tf.layers.batchNormalization().apply(v) as tf.SymbolicTensor;
  v = tf.layers.reLU().apply(v) as tf.SymbolicTensor;
  v = tf.layers.flatten().apply(v) as tf.SymbolicTensor;
  
  v = tf.layers.dense({
    units: 64,
    activation: 'relu'
  }).apply(v) as tf.SymbolicTensor;
  
  // Tanh for value (-1 to 1)
  const valueOutput = tf.layers.dense({
    units: 1,
    activation: 'tanh',
    name: 'value_head'
  }).apply(v) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });
  
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: ['categoricalCrossentropy', 'meanSquaredError']
  });

  return model;
};
