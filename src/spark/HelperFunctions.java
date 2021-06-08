package spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;

public class HelperFunctions {

    /**
     * Calculates the min from a list of numbers
     * @param list
     * @param <T extends Number>
     * @return
     */
    private static <T extends Number> double getMin(List<T> list) {
        return list.stream().mapToDouble(Number::doubleValue).min().orElse(0);
    }

    /**
     * Calculates the max from a list of numbers
     * @param list
     * @param <T extends Number>
     * @return
     */
    private static <T extends Number> double getMax(List<T> list) {
        return list.stream().mapToDouble(Number::doubleValue).max().orElse(0);
    }

    /**
     * Calculates the average from a list of numbers
     * @param list
     * @param <T extends Number>
     * @return
     */
    private static <T extends Number> double getAverage(List<T> list) {
        return list.stream()
                .mapToDouble(Number::doubleValue)
                .average()
                .orElse(0);
    }

    /**
     * A function to calculate the sample standard deviation from a list of doubles
     * https://www.programiz.com/java-programming/examples/standard-deviation
     * @param values
     * @return Sample standard deviation
     */
    private static double calculateSD(List<Double> values)
    {
        Double[] numArray = values.toArray(new Double[0]);

        double sum = 0.0, standardDeviation = 0.0;
        int length = numArray.length;

        for(double num : numArray) {
            sum += num;
        }

        double mean = sum/length;

        for(double num: numArray) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation/(length-1));
    }

    /**
     * Creates a spark dataset containing the data from an unlabelled CSV file.
     *
     * @param spark - the current SparkSession
     * @param filePath - path to the input CSV file
     * @return a Dataset<Row> containing the CSV data
     */
    public static Dataset<Row> createDatasetFromCSV(SparkSession spark, String filePath) {

        JavaRDD<String> lines = spark.sparkContext().textFile(filePath,0).toJavaRDD();

        JavaRDD<LabeledPoint> linesRDD = lines.map(line ->{
            String[] tokens = line.split(",");
            double[] features = new double[tokens.length - 1];
            for (int i = 0; i < features.length; i++) {
                features[i] = Double.parseDouble(tokens[i]);
            }
            Vector v = new DenseVector(features);

            if(tokens[features.length].equals("anomaly")) {
                return new LabeledPoint(0.0, v);
            } else {
                return new LabeledPoint(1.0, v);
            }

        });

        return spark.createDataFrame(linesRDD, LabeledPoint.class);
    }

    private enum ResultType {
        TRAINING,
        TESTING,
        TIMING
    }

    private static void printResults(ResultType resultType, double average, double min, double max, double stdDev) {
        System.out.println("\n");
        System.out.println(resultType + ":");
        System.out.println("Average: " + average);
        System.out.println("Min: " + min);
        System.out.println("Max: " + max);
        if (stdDev != -1) {
            System.out.println("Standard Deviation: " + stdDev);
        }
    }

    public static void printAllResults(List<Double> trainAccuracies, List<Double> testAccuracies, List<Long> runningTimes) {

        // TRAINING ACCURACY
        double averageTrainAccuracy = getAverage(trainAccuracies);
        double minTrainAccuracy = getMin(trainAccuracies);
        double maxTrainAccuracy = getMax(trainAccuracies);
        double trainingStandardDeviation = calculateSD(trainAccuracies);

        // TESTING ACCURACY
        double averageTestAccuracy = getAverage(testAccuracies);
        double minTestAccuracy = getMin(testAccuracies);
        double maxTestAccuracy = getMax(testAccuracies);
        double testingStandardDeviation = calculateSD(testAccuracies);

        // TIMING
        double averageRunningTime = getAverage(runningTimes);
        double minRunningTime = getMin(runningTimes);
        double maxRunningTime = getMax(runningTimes);

        System.out.println("\n\n=====================================================");

        printResults(ResultType.TRAINING,
                averageTrainAccuracy,
                minTrainAccuracy,
                maxTrainAccuracy,
                trainingStandardDeviation);

        printResults(ResultType.TESTING,
                averageTestAccuracy,
                minTestAccuracy,
                maxTestAccuracy,
                testingStandardDeviation);

        printResults(ResultType.TIMING,
                averageRunningTime,
                minRunningTime,
                maxRunningTime,
                -1);
    }

}
