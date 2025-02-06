import java.util.Arrays;

public class Main {

    public Main() {
        Neuron n00 = new Neuron(new double[] { -0.52, -1.492 }, 0.649);
        Neuron n01 = new Neuron(new double[] { 3.347, -3.553 }, -3.568);
        Neuron n02 = new Neuron(new double[] { 15.765, -15.805 }, 0.47);

        Neuron n10 = new Neuron(new double[] { -0.035, 35.234, -15.693 }, 4.349);

        Layer[] layers = new Layer[2];
        layers[0] = new Layer(new Neuron[] { n00, n01, n02 });
        layers[1] = new Layer(new Neuron[] { n10 });

        NeuralNet neuralNet = new NeuralNet(layers);

        double[] test = neuralNet.predict(new double[] { 1, 0 });
        System.out.println(Arrays.toString(test));

        double[] test1 = neuralNet.predict(new double[] { 0, 0 });
        System.out.println(Arrays.toString(test1));
    }

    public static void main(String[] args) throws Exception {
        new Main();
    }

}
