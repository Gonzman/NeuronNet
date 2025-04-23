package neuralnet;

import java.io.Serializable;

import neuralnet.mode.Mode;
import neuralnet.mode.Softmax;

public class Layer implements Serializable {

    private Neuron[] neurons;
    private Mode mode;
    private double[] lastInputs;
    private double[] preActivations;

    public Layer(Mode mode, Neuron ...neurons) {
        this.neurons = neurons;
        this.mode = mode;
    }

    public Layer(Mode mode, int numNeurons, int numInputs) {
        this.neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            double[] weights = new double[numInputs];
            for (int j = 0; j < numInputs; j++) {
                weights[j] = Math.random() * 2 - 1;
            }
            double bias = Math.random() * 2 - 1;
            neurons[i] = new Neuron(weights, bias);
        }
        this.mode = mode;
    }

    public double[] forward(double[] inputs) {
        this.lastInputs = inputs.clone(); // Speichern für Backpropagation
        
        preActivations = new double[neurons.length];
        double[] outputs = new double[neurons.length];
        
        // Berechne erstmal die rohen Neuron-Ausgaben
        for (int i = 0; i < neurons.length; i++) {
            preActivations[i] = neurons[i].computePreActivation(inputs);
            
            if (!(mode instanceof Softmax)) {
                // Aktivierungsfunktion direkt anwenden (nicht für Softmax)
                outputs[i] = mode.compute(preActivations[i]);
            }
        }
        
        // Softmax braucht Spezialbehandlung
        if (mode instanceof Softmax) {
            // Wende Softmax auf alle Outputs zusammen an
            double max = Double.NEGATIVE_INFINITY;
            for (double val : preActivations) {
                if (val > max) {
                    max = val;
                }
            }
            
            double sum = 0.0;
            for (int i = 0; i < preActivations.length; i++) {
                // Max abziehen für numerische Stabilität
                outputs[i] = Math.exp(preActivations[i] - max);
                sum += outputs[i];
            }
            
            // Normalisieren für Wahrscheinlichkeiten
            for (int i = 0; i < outputs.length; i++) {
                outputs[i] /= sum;
            }
        }
        return outputs;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        // Initialisiere Gradient für Inputs dieser Schicht
        double[] inputGradient = new double[neurons[0].getWeights().length];
        
        double[] deltas = new double[neurons.length];
        
        // Softmax-Ableitung anders behandeln
        if (mode instanceof Softmax) {
            // Bei Cross-Entropy mit Softmax ist Delta vereinfacht
            // zu (vorhergesagt - ziel), was unser outputGradient schon ist
            deltas = outputGradient;
        } else {
            // Für andere Aktivierungsfunktionen normale Ableitung berechnen
            for (int i = 0; i < neurons.length; i++) {
                deltas[i] = outputGradient[i] * mode.derivative(preActivations[i]);
            }
        }
        
        // Verarbeite jedes Neuron in der Schicht
        for (int i = 0; i < neurons.length; i++) {
            double delta = deltas[i];
            
            double[] weights = neurons[i].getWeights();
            
            // Update jedes Gewicht des Neurons
            for (int j = 0; j < weights.length; j++) {
                // Füge zum Input-Gradienten hinzu (für vorherige Schicht)
                inputGradient[j] += delta * weights[j];
                
                // Gewicht aktualisieren: w = w + Lernrate * delta * input
                weights[j] += learningRate * delta * lastInputs[j];
            }
            
            // Bias aktualisieren: b = b + Lernrate * delta
            neurons[i].setBias(neurons[i].getBias() + learningRate * delta);
        }
        
        return inputGradient;
    }

    @Override
    public String toString() {
        int numNeurons = neurons.length;
        String str = getClass().getSimpleName() + "(";
        if (numNeurons <= 5) {
            for (Neuron neuron : neurons) {
                str += "\n\t" + neuron;
            }
        } else {
            for (int i = 0; i < 3; i++) {
                str += "\n\t" + neurons[i];
            }
            str += "\n\t...\n\t" + neurons[numNeurons - 1];
        }
        return str + "\n)";
    }
}
