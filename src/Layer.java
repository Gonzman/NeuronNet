import java.io.Serializable;

import mode.Mode;

public class Layer implements Serializable {

    private Neuron[] neurons;
    private Mode mode;

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
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs, mode);
        }
        return outputs;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        double[] inputGradient = new double[neurons[0].getWeights().length];
        double[] neuronGradients = new double[neurons.length];
        
        for (int i = 0; i < neurons.length; i++) {
            neuronGradients[i] = outputGradient[i] * mode.derivative(neurons[i].getBias());
        }
        
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = neurons[i].getWeights();
            for (int j = 0; j < weights.length; j++) {
                inputGradient[j] += neuronGradients[i] * weights[j];
                weights[j] += learningRate * neuronGradients[i];
            }
            neurons[i].setBias(neurons[i].getBias() + learningRate * neuronGradients[i]);
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
