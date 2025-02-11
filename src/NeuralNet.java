import java.util.ArrayList;

public class NeuralNet {

    private ArrayList<Layer> layers = new ArrayList<Layer>();

    public NeuralNet() {
        
    }

    public double[] predict(double[] pinputs) {
        double[] inputs = pinputs;
        for (int i = 0; i < layers.size(); i++) {
            inputs = layers.get(i).forward(inputs);
        }
        return inputs;
    }

    public void addLayer(Layer layer){
        layers.add(layer);
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
