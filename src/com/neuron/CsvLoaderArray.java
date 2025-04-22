import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class CsvLoaderArray {

    public static void main(String[] args) {
        String file = "src/test_set.csv"; // <-- Make sure the path is correct
        List<double[][]> data = loadDataAsArrays(file);
        if (data != null) {
            System.out.println("Data loaded successfully.");
        }
    }

    public static List<double[][]> loadDataAsArrays(String file) {
        int numInputFeatures = 7;
        int numClasses = 7;

        int rowCount = 0;

        // --- First Pass: Count valid data rows ---
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            boolean isHeader = true;
            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                if (!line.trim().isEmpty()) {
                    rowCount++;
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading the file: " + file);
            e.printStackTrace();
            return null;
        }

        if (rowCount == 0) {
            System.err.println("No data rows found in the file.");
            return null;
        }

        double[][] inputs = new double[rowCount][numInputFeatures];
        double[][] targets = new double[rowCount][numClasses];

        // --- Second Pass: Read and populate arrays ---
        int currentIndex = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            boolean isHeader = true;

            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue;
                }

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] values = line.split(",");

                if (values.length < 9) {
                    System.err.println("Skipping line " + (currentIndex + 1) + ": Not enough columns.");
                    continue;
                }

                try {
                    for (int i = 0; i < numInputFeatures; i++) {
                        inputs[currentIndex][i] = Double.parseDouble(values[i].trim());
                    }

                    int rasseCode = Integer.parseInt(values[8].trim());
                    if (rasseCode >= 0 && rasseCode < numClasses) {
                        targets[currentIndex][rasseCode] = 1.0;
                    } else {
                        System.err.println("Invalid Rasse_Code: " + rasseCode + " at row " + (currentIndex + 1));
                    }

                    currentIndex++;

                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                    System.err.println("Skipping row due to parsing error: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading the data: " + file);
            e.printStackTrace();
            return null;
        }

        // If fewer rows were processed, trim the arrays
        if (currentIndex < rowCount) {
            inputs = Arrays.copyOf(inputs, currentIndex);
            targets = Arrays.copyOf(targets, currentIndex);
            System.out.println("Trimmed arrays to " + currentIndex + " valid rows.");
        }

        return Arrays.asList(inputs, targets);
    }
}
