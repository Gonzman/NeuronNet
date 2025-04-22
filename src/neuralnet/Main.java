package neuralnet;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import neuralnet.mode.LeakyReLu;
import neuralnet.mode.ReLu;
import neuralnet.mode.Sigmoid;
import neuralnet.mode.TanH;

public class Main {
    public static void main(String[] args) {
        Random random = new Random();

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
            {1},
            {1},
            {0}
        };

        neuralNet.train(inputs, targets, 0.1, 1000);
        
        System.out.println(Arrays.toString(neuralNet.predict(new double[] {1, 1})));
        
    }
}
