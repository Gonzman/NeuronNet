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

    public void addLayer(Layer ...layers){
        for (Layer layer : layers) {
            this.layers.add(layer);
        }
    }

    public void train(double[][] inputs, double[][] targets, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < inputs.length; i++) {
                double[] output = predict(inputs[i]);
                
                // Compute loss (Mean Squared Error)
                double[] error = new double[output.length];
                double loss = 0.0;
                for (int j = 0; j < output.length; j++) {
                    error[j] = targets[i][j] - output[j];
                    loss += Math.pow(error[j], 2);
                }
                totalLoss += loss / output.length;
                
                // Backpropagation
                double[] gradient = error; // Assuming last layer uses error directly
                for (int j = layers.size() - 1; j >= 0; j--) {
                    gradient = layers.get(j).backward(gradient, learningRate);
                }
            }
            
            // Print loss for monitoring
            System.out.println("Epoch " + (epoch + 1) + " Loss: " + (totalLoss / inputs.length));
        }

        System.out.println(this.toString());
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
