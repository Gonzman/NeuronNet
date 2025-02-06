public class Neuron {
    private double bias;
    private double[] weights;

    public Neuron(double[] weigts, double bias){
        this.weights = weigts;
        this.bias = bias;
    }

    public double forward(double[] input){
        double value = bias;
        for (int i = 0; i < input.length; i++) {
            value += input[i]*weights[i];
        }

        return 1/(1+Math.exp(-value));
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

}
