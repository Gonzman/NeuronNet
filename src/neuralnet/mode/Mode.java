package neuralnet.mode;

import java.io.Serializable;

public interface Mode extends Serializable {
    public double compute(double input);
    public double derivative(double input);
}
