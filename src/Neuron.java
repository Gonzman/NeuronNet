public class Neuron {

    private double bias;
    private double[] weights;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double forward(double[] input) {
        double value = bias;
        for (int i = 0; i < input.length; i++) {
            value += input[i] * weights[i];
        }
        return 1 / (1 + Math.exp(-value));
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

    public void setWeigts(double[] weights) {
        this.weights = weights;
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
