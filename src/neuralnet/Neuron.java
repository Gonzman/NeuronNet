package neuralnet;

import java.io.Serializable;

import neuralnet.mode.Mode;

public class Neuron implements Serializable {

    private double bias;
    private double[] weights;
    private double[] lastInputs; // Store last inputs for backpropagation
    private double lastOutput; // Store last output for backpropagation
    private double preActivation; // Store the value before activation function

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double forward(double[] input, Mode mode) {
        this.lastInputs = input.clone(); // Store inputs for backpropagation
        
        preActivation = bias;
        for (int i = 0; i < input.length; i++) {
            preActivation += input[i] * weights[i];
        }
        lastOutput = mode.compute(preActivation);
        return lastOutput;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }
    
    public double getLastOutput() {
        return lastOutput;
    }
    
    public double[] getLastInputs() {
        return lastInputs;
    }
    
    public double getPreActivation() {
        return preActivation;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        double[] inputGradient = new double[weights.length];
        
        // Compute gradients for weights and bias
        for (int i = 0; i < weights.length; i++) {
            inputGradient[i] = outputGradient[i] * weights[i];
            weights[i] += learningRate * outputGradient[i];
        }
        
        // Update bias
        bias += learningRate * outputGradient[0];
        
        return inputGradient;
    }

    @Override
    public String toString() {
        int numInputs = weights.length;
        String str = getClass().getSimpleName() + "( weights=[ ";
        if (numInputs <= 5) {
            for (int i = 0; i < numInputs; i++) {
                str += weights[i];
                if (i != numInputs - 1) {
                    str += ", ";
                }
            }
        } else {
            for (int i = 0; i < 3; i++) {
                str += weights[i] + ", ";
            }
            str += "..., " + weights[numInputs - 1];
        }
        return str + " ], bias=" + bias + " )";
    }

}
