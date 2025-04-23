package neuralnet.mode;

public class Softmax implements Mode {
    @Override
    public double compute(double input) {
        // For the Softmax, this is handled differently as we need all outputs
        // This just returns the input as Softmax will be applied at the layer level
        return input;
    }
    
    @Override
    public double derivative(double input) {
        // The derivative is handled differently for Softmax with cross-entropy loss
        // But for simplicity, we'll return 1 here
        return 1.0;
    }
}