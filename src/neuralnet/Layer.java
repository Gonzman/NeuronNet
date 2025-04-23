package neuralnet;

import java.io.Serializable;

import neuralnet.mode.Mode;
import neuralnet.mode.Softmax;

public class Layer implements Serializable {

    private Neuron[] neurons;
    private Mode mode;
    private double[] lastInputs;
    private double[] preActivations;

    public Layer(Mode mode, Neuron ...neurons) {
        this.neurons = neurons;
        this.mode = mode;
    }

    public Layer(Mode mode, int numNeurons, int numInputs) {
        this.neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            double[] weights = new double[numInputs];
            for (int j = 0; j < numInputs; j++) {
                weights[j] = Math.random() * 2 - 1;
            }
            double bias = Math.random() * 2 - 1;
            neurons[i] = new Neuron(weights, bias);
        }
        this.mode = mode;
    }

    public double[] forward(double[] inputs) {
        this.lastInputs = inputs.clone(); // Store for backpropagation
        
        preActivations = new double[neurons.length];
        double[] outputs = new double[neurons.length];
        
        // First compute the raw neuron outputs
        for (int i = 0; i < neurons.length; i++) {
            preActivations[i] = neurons[i].computePreActivation(inputs);
            
            if (!(mode instanceof Softmax)) {
                // Apply activation function directly for non-softmax layers
                outputs[i] = mode.compute(preActivations[i]);
            }
        }
        
        // Handle Softmax activation function specially
        if (mode instanceof Softmax) {
            // Apply softmax to all outputs together
            double max = Double.NEGATIVE_INFINITY;
            for (double val : preActivations) {
                if (val > max) {
                    max = val;
                }
            }
            
            double sum = 0.0;
            for (int i = 0; i < preActivations.length; i++) {
                // Subtract max for numerical stability
                outputs[i] = Math.exp(preActivations[i] - max);
                sum += outputs[i];
            }
            
            // Normalize to get probabilities
            for (int i = 0; i < outputs.length; i++) {
                outputs[i] /= sum;
            }
        }
        return outputs;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        // Initialize gradient for inputs to this layer
        double[] inputGradient = new double[neurons[0].getWeights().length];
        
        double[] deltas = new double[neurons.length];
        
        // Handle softmax derivative differently
        if (mode instanceof Softmax) {
            // For cross-entropy loss with softmax, the delta is simplified
            // to (predicted - target) which is the outputGradient we already have
            deltas = outputGradient;
        } else {
            // For other activation functions, calculate normal derivative
            for (int i = 0; i < neurons.length; i++) {
                deltas[i] = outputGradient[i] * mode.derivative(preActivations[i]);
            }
        }
        
        // Process each neuron in the layer
        for (int i = 0; i < neurons.length; i++) {
            double delta = deltas[i];
            
            double[] weights = neurons[i].getWeights();
            
            // Update each weight of the neuron
            for (int j = 0; j < weights.length; j++) {
                // Add to the input gradient (for previous layer)
                inputGradient[j] += delta * weights[j];
                
                // Update weight: w = w + learning_rate * delta * input
                weights[j] += learningRate * delta * lastInputs[j];
            }
            
            // Update bias: b = b + learning_rate * delta
            neurons[i].setBias(neurons[i].getBias() + learningRate * delta);
        }
        
        return inputGradient;
    }

    @Override
    public String toString() {
        int numNeurons = neurons.length;
        String str = getClass().getSimpleName() + "(";
        if (numNeurons <= 5) {
            for (Neuron neuron : neurons) {
                str += "\n\t" + neuron;
            }
        } else {
            for (int i = 0; i < 3; i++) {
                str += "\n\t" + neurons[i];
            }
            str += "\n\t...\n\t" + neurons[numNeurons - 1];
        }
        return str + "\n)";
    }
}
