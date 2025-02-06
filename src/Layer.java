public class Layer {
    private Neuron[] neurons;
    public Layer(Neuron[] neurons){
        this.neurons = neurons;
    }

    public double[] forward(double[] inputs){
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs);
        }
        return outputs;
    }

    @Override
    public String toString() {
        
        return super.toString();
    }

}
