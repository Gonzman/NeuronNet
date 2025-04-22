package neuralnet.mode;

public class TanH implements Mode {
    @Override
    public double compute(double input) {
        return Math.tanh(input);
    }
    
    @Override
    public double derivative(double input) {
        double tanh = compute(input);
        return 1 - (tanh * tanh);
    }
}