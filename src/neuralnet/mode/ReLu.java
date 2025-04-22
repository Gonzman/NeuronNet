package neuralnet.mode;

public class ReLu implements Mode{

    @Override
    public double compute(double input) {
        if(input <= 0){
            return 0;
        }
        return input;
    }

    public double derivative(double input) {
        return input > 0 ? 1 : 0;
    }
    
}
