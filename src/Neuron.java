import mode.Mode;

public class Neuron {

    private double bias;
    private double[] weights;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double forward(double[] input, Mode mode) {
        double value = bias;
        for (int i = 0; i < input.length; i++) {
            value += input[i] * weights[i];
        }
        return mode.compute(value);
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
