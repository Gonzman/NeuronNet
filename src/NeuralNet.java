public class NeuralNet {
    private Layer[] layers;
    
    public NeuralNet(Layer[] layers){
        this.layers = layers;
    }

    public double[] predict(double[] pinputs){
        double[] inputs = pinputs;
        for (int i = 0; i < layers.length; i++) {
            inputs = layers[i].forward(inputs);
        }

        return inputs;
    }
}
