package mode;

public class Sigmoid implements Mode{

    @Override
    public double compute(double input) {
        return 1 / (1 + Math.exp(-input));
    }
    
}
