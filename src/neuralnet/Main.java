package neuralnet;

import java.util.List;

import neuralnet.mode.LeakyReLu;
import neuralnet.mode.Softmax;

public class Main {
    public static void main(String[] args) {

        NeuralNet neuralNet = new NeuralNet();
        
        // Configure neural network for dog breed classification (7 classes)
        neuralNet.addLayer(new Layer(new LeakyReLu(0.01), 7, 7));  // 7 neurons, 7 inputs
        neuralNet.addLayer(new Layer(new LeakyReLu(0.01), 14, 7)); // Hidden layer with more neurons
        neuralNet.addLayer(new Layer(new Softmax(), 7, 14));      // Output layer with 7 neurons (one per breed)

        System.out.println("Lade Trainingsdaten...");
        List<double[][]> trainData = CsvLoaderArray.loadDataAsArrays("src/neuralnet/train_set.csv");
        double[][] trainInputs = trainData.get(0);
        double[][] trainTargets = trainData.get(1);
        
        // Normalisiert die Eingabedaten, um NaN-Werte zu vermeiden
        double[] means = calculateMeans(trainInputs);
        double[] stdDevs = calculateStdDevs(trainInputs);
        double[][] normalizedTrainInputs = normalizeInputs(trainInputs, means, stdDevs);
        
        System.out.println("Trainiere neuronales Netz für die Klassifikation von Hunderassen...");
        // Trainiert mit einer Lernrate, die für Mehrklassenklassifikation angepasst ist
        neuralNet.train(normalizedTrainInputs, trainTargets, 0.002, 1000);
        
        // Lädt und normalisiert Testdaten
        System.out.println("\nBewertung auf dem Testset...");
        List<double[][]> testData = CsvLoaderArray.loadDataAsArrays("src/neuralnet/test_set.csv");
        double[][] testInputs = testData.get(0);
        double[][] testTargets = testData.get(1);
        
        double[][] normalizedTestInputs = normalizeInputs(testInputs, means, stdDevs);
        
        // Bewertung auf dem Testset
        int correct = 0;
        int total = normalizedTestInputs.length;
        
        // Definiert Rassennamen, um die Ausgabe aussagekräftiger zu machen
        String[] breedNames = {
            "Schäferhund", 
            "Dackel",
            "Dobermann",
            "Chihuahua",
            "Bernhardiner",
            "Collie",
            "Mops"
        };
        
        for (int i = 0; i < total; i++) {
            double[] prediction = neuralNet.predict(normalizedTestInputs[i]);
            
            // Findet den Index des höchsten Wertes im Vorhersage-Array (wahrscheinlichste Rasse)
            int predictedBreedIdx = findMaxIndex(prediction);
            
            // Findet die tatsächliche Rasse (Index des Wertes 1.0 im Ziel-One-Hot-Encoded-Array)
            int actualBreedIdx = findMaxIndex(testTargets[i]);
            
            if (predictedBreedIdx == actualBreedIdx) {
                correct++;
            }
            
            System.out.printf("Beispiel %d: Vorhergesagte Rasse: %s (%.2f), Tatsächliche Rasse: %s, %s%n", 
                    i, 
                    breedNames[predictedBreedIdx], 
                    prediction[predictedBreedIdx],
                    breedNames[actualBreedIdx], 
                    predictedBreedIdx == actualBreedIdx ? "RICHTIG" : "FALSCH");
        }
        
        double accuracy = (double) correct / total * 100;
        System.out.printf("\nTestgenauigkeit: %.2f%% (%d/%d richtig)%n", 
                accuracy, correct, total);

        
        double[] test = neuralNet.predict(normalizeInput(new double[] {
                    72.73797123480357, 40.369457700466384,0,36.96701940040313,63.76767217091086,70.88369928087936,3.522749285654476
                }, means, stdDevs));

        int predictedBreedIdx = findMaxIndex(test);
        System.out.println("Vorhergesagte Rasse: " + breedNames[predictedBreedIdx] + " mit Konfidenz: " + 
                String.format("%.2f", test[predictedBreedIdx]));

        // Gibt alle Konfidenzwerte aus
        System.out.println("\nAlle Konfidenzwerte:");
        for (int i = 0; i < test.length; i++) {
            System.out.println(breedNames[i] + ": " + String.format("%.2f", test[i]));
        }
    }
    
    // Hilfsmethode, um den Index des maximalen Wertes in einem Array zu finden
    private static int findMaxIndex(double[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    // Normalisiert alle Eingabeproben
    private static double[][] normalizeInputs(double[][] inputs, double[] means, double[] stdDevs) {
        double[][] normalized = new double[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            normalized[i] = normalizeInput(inputs[i], means, stdDevs);
        }
        return normalized;
    }
    
    // Normalisiert eine einzelne Eingabe
    private static double[] normalizeInput(double[] input, double[] means, double[] stdDevs) {
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            if (stdDevs[i] == 0) {
                normalized[i] = 0; // Vermeidet Division durch Null
            } else {
                normalized[i] = (input[i] - means[i]) / stdDevs[i];
            }
        }
        return normalized;
    }
    
    // Berechnet Mittelwerte für jeden Feature
    private static double[] calculateMeans(double[][] inputs) {
        int numFeatures = inputs[0].length;
        double[] means = new double[numFeatures];
        
        for (int i = 0; i < numFeatures; i++) {
            double sum = 0;
            for (double[] input : inputs) {
                sum += input[i];
            }
            means[i] = sum / inputs.length;
        }
        
        return means;
    }
    
    // Berechnet Standardabweichungen für jeden Feature
    private static double[] calculateStdDevs(double[][] inputs) {
        int numFeatures = inputs[0].length;
        double[] means = calculateMeans(inputs);
        double[] stdDevs = new double[numFeatures];
        
        for (int i = 0; i < numFeatures; i++) {
            double sumSquaredDiff = 0;
            for (double[] input : inputs) {
                double diff = input[i] - means[i];
                sumSquaredDiff += diff * diff;
            }
            stdDevs[i] = Math.sqrt(sumSquaredDiff / inputs.length);
            // Verhindert Division durch Null 
            if (stdDevs[i] < 0.0001) {
                stdDevs[i] = 1.0;
            }
        }
        
        return stdDevs;
    }
}
