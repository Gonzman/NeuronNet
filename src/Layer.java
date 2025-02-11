import mode.Mode;

public class Layer {

    private Neuron[] neurons;
    private Mode mode;

    public Layer(Neuron[] neurons, Mode mode) {
        this.neurons = neurons;
        this.mode = mode;
    }

    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs, mode);
        }
        return outputs;
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
