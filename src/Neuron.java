public class Neuron {
    private double bias;
    private double[] weigts;

    public Neuron(double[] weigts, double bias){
        this.weigts = weigts;
        this.bias = bias;
    }

    public double forward(double[] input){
        double value = bias;
        for (int i = 0; i < input.length; i++) {
            value += input[i]*weigts[i];
        }

        return 1/(1+Math.exp(value*-1));
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double[] getWeigts() {
        return weigts;
    }

    public void setWeigts(double[] weigts) {
        this.weigts = weigts;
    }

}
