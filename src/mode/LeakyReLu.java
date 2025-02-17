package mode;

public class LeakyReLu implements Mode{

    private double alpha;

    public LeakyReLu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double compute(double input) {
        if(input <= 0){
            return input * alpha;
        }
        return input;
    }

    public double derivative(double input) {
        return input > 0 ? 1 : alpha;
    }
    
}
