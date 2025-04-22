package neuralnet;

import java.io.Serializable;

import neuralnet.mode.Mode;

public class Layer implements Serializable {

    private Neuron[] neurons;
    private Mode mode;
    private double[] lastInputs;

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
        
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs, mode);
        }
        return outputs;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        // Initialize gradient for inputs to this layer
        double[] inputGradient = new double[neurons[0].getWeights().length];
        
        // Process each neuron in the layer
        for (int i = 0; i < neurons.length; i++) {
            // Calculate gradient for this neuron's output
            // δ = outputGradient * derivative of activation function
            double delta = outputGradient[i] * mode.derivative(neurons[i].getPreActivation());
            
            double[] weights = neurons[i].getWeights();
            double[] inputs = neurons[i].getLastInputs();
            
            // Update each weight of the neuron
            for (int j = 0; j < weights.length; j++) {
                // Add to the input gradient (for previous layer)
                inputGradient[j] += delta * weights[j];
                
                // Update weight: w = w + learning_rate * delta * input
                weights[j] += learningRate * delta * inputs[j];
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
