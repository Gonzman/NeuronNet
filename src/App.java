import java.util.Arrays;

import mode.Sigmoid;

public class App {
    public static void main(String[] args) throws Exception {
        Neuron n00 = new Neuron(new double[] { -0.52, -1.492 }, 0.649);
        Neuron n01 = new Neuron(new double[] { 3.347, -3.553 }, -3.568);
        Neuron n02 = new Neuron(new double[] { 15.765, -15.805 }, 0.47);

        Neuron n10 = new Neuron(new double[] { -0.035, 35.234, -15.693 }, 4.349);

        NeuralNet neuralNet = new NeuralNet();
        neuralNet.addLayer(new Layer(new Sigmoid(), n00, n01, n02 ));
        neuralNet.addLayer(new Layer(new Sigmoid(),n10));

        double[] test = neuralNet.predict(new double[] { 1, 0 });
        System.out.println(Arrays.toString(test));

        double[] test1 = neuralNet.predict(new double[] { 0, 0 });
        System.out.println(Arrays.toString(test1));
        
    }
}
