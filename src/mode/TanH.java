package mode;

public class TanH implements Mode{

    @Override
    public double compute(double input) {
        return Math.tanh(input);
    }
    
}
