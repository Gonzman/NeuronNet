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
        neuralNet.addLayer(new Layer(new LeakyReLu(0.01), 14, 14)); // Another hidden layer for more complexity
        neuralNet.addLayer(new Layer(new Softmax(), 7, 14));      // Output layer with 7 neurons (one per breed)

        System.out.println("Loading training data...");
        List<double[][]> trainData = CsvLoaderArray.loadDataAsArrays("src/neuralnet/train_set.csv");
        double[][] trainInputs = trainData.get(0);
        double[][] trainTargets = trainData.get(1);
        
        // Normalize the input data to prevent NaN values
        double[] means = calculateMeans(trainInputs);
        double[] stdDevs = calculateStdDevs(trainInputs);
        double[][] normalizedTrainInputs = normalizeInputs(trainInputs, means, stdDevs);
        
        System.out.println("Training neural network for dog breed classification...");
        // Train with a learning rate adjusted for multi-class classification
        neuralNet.train(normalizedTrainInputs, trainTargets, 0.002, 1000);
        
        // Load and normalize test data
        System.out.println("\nEvaluating on test set...");
        List<double[][]> testData = CsvLoaderArray.loadDataAsArrays("src/neuralnet/test_set.csv");
        double[][] testInputs = testData.get(0);
        double[][] testTargets = testData.get(1);
        
        double[][] normalizedTestInputs = normalizeInputs(testInputs, means, stdDevs);
        
        // Evaluate on test set
        int correct = 0;
        int total = normalizedTestInputs.length;
        
        // Define breed names to make output more meaningful
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
            
            // Find the index of the highest value in the prediction array (most likely breed)
            int predictedBreedIdx = findMaxIndex(prediction);
            
            // Find the actual breed (index of value 1.0 in the target one-hot encoded array)
            int actualBreedIdx = findMaxIndex(testTargets[i]);
            
            if (predictedBreedIdx == actualBreedIdx) {
                correct++;
            }
            
            System.out.printf("Sample %d: Predicted breed: %s (%.2f), Actual breed: %s, %s%n", 
                    i, 
                    breedNames[predictedBreedIdx], 
                    prediction[predictedBreedIdx],
                    breedNames[actualBreedIdx], 
                    predictedBreedIdx == actualBreedIdx ? "CORRECT" : "WRONG");
        }
        
        double accuracy = (double) correct / total * 100;
        System.out.printf("\nTest Accuracy: %.2f%% (%d/%d correct)%n", 
                accuracy, correct, total);

        
        double[] test = neuralNet.predict(normalizeInput(new double[] {
                    72.73797123480357, 40.369457700466384,0,36.96701940040313,63.76767217091086,70.88369928087936,3.522749285654476
                }, means, stdDevs));

        int predictedBreedIdx = findMaxIndex(test);
        System.out.println("Predicted breed: " + breedNames[predictedBreedIdx] + " with confidence: " + 
                String.format("%.2f", test[predictedBreedIdx]));

        // Print all confidence values
        System.out.println("\nAll confidence values:");
        for (int i = 0; i < test.length; i++) {
            System.out.println(breedNames[i] + ": " + String.format("%.2f", test[i]));
        }
    }
    
    // Helper method to find the index of the maximum value in an array
    private static int findMaxIndex(double[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    // Normalize all input samples
    private static double[][] normalizeInputs(double[][] inputs, double[] means, double[] stdDevs) {
        double[][] normalized = new double[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            normalized[i] = normalizeInput(inputs[i], means, stdDevs);
        }
        return normalized;
    }
    
    // Normalize a single input
    private static double[] normalizeInput(double[] input, double[] means, double[] stdDevs) {
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            if (stdDevs[i] == 0) {
                normalized[i] = 0; // Avoid division by zero
            } else {
                normalized[i] = (input[i] - means[i]) / stdDevs[i];
            }
        }
        return normalized;
    }
    
    // Calculate means for each feature
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
    
    // Calculate standard deviations for each feature
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
            // Prevent division by zero in normalization step
            if (stdDevs[i] < 0.0001) {
                stdDevs[i] = 1.0;
            }
        }
        
        return stdDevs;
    }
}
