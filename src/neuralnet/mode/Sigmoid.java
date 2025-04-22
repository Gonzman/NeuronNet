package neuralnet.mode;

public class Sigmoid implements Mode {
    @Override
    public double compute(double input) {
        return 1 / (1 + Math.exp(-input));
    }
    
    @Override
    public double derivative(double input) {
        double sigmoid = compute(input);
        return sigmoid * (1 - sigmoid);
    }
}