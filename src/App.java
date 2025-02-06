public class App {
    public static void main(String[] args) throws Exception {

        Neuron n00 = new Neuron(new double[]{-0.52,-1.492},0.649);
        Neuron n01 = new Neuron(new double[]{3.347,-3.553},-3.568);
        Neuron n02 = new Neuron(new double[]{15.765,-15.805},0.47);

        
        Neuron n10 = new Neuron(new double[]{-0.035,35.234,-15.693},4.349);
        
        Layer[] layers = new Layer[2];
        layers[0] = new Layer(new Neuron[]{n00,n01,n02});
        layers[1] = new Layer(new Neuron[]{n10});

        NeuralNet neuralNet = new NeuralNet(layers);
        var test = neuralNet.predict(new double[]{1,0});
        for (double d : test) {
            System.out.println(d);
        }
        
        var test1 = neuralNet.predict(new double[]{1,1});
        for (double d : test1) {
            System.out.println(d);
        }
        
    }
}
