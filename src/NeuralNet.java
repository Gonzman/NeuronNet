public class NeuralNet {

    private Layer[] layers;

    public NeuralNet(Layer[] layers) {
        this.layers = layers;
    }

    public double[] predict(double[] pinputs) {
        double[] inputs = pinputs;
        for (int i = 0; i < layers.length; i++) {
            inputs = layers[i].forward(inputs);
        }
        return inputs;
    }

    @Override
    public String toString() {
        String str = getClass().getSimpleName() + "(";
        for (Layer layer : layers) {
            str += "\n\t" + layer.toString().replace("\n", "\n\t");
        }
        return str + "\n)";
    }

}
