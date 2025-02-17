import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import mode.LeakyReLu;
import mode.ReLu;
import mode.Sigmoid;
import mode.TanH;

public class Main {
    public static void main(String[] args) {
        Random random = new Random();

        Neuron n00 = new Neuron(new double[] { random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1 }, random.nextDouble() * 2 - 1);
        Neuron n01 = new Neuron(new double[] { random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1 }, random.nextDouble() * 2 - 1);
        Neuron n02 = new Neuron(new double[] { random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1 }, random.nextDouble() * 2 - 1);

        Neuron n10 = new Neuron(new double[] { random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1 }, random.nextDouble() * 2 - 1);

        NeuralNet neuralNet = new NeuralNet();
        //neuralNet.addLayer(new Layer(new Sigmoid(), n00, n01, n02 ));
        //neuralNet.addLayer(new Layer(new Sigmoid(), n10));
        neuralNet.addLayer(new Layer(new LeakyReLu(0.1), 3, 2));
        neuralNet.addLayer(new Layer(new LeakyReLu(0.1), 1, 3));
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] targets = {
            {0},
            {0},
            {0},
            {1}
        };

        neuralNet.train(inputs, targets, 0.1, 1000);
        
        System.out.println(Arrays.toString(neuralNet.predict(new double[] {1, 1})));
        
    }
}
