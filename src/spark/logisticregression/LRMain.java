package spark.logisticregression;

import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static spark.HelperFunctions.*;


public class LRMain {

    public static void main(String[] args) {

        disableLogging();

        final String appName = "SparkLogisticRegression";
        final String inputDir = args[0];
        final String outputDir = args[1];
        final int NUM_ITERATIONS = 10;

        SparkSession spark = SparkSession
                .builder()
                .appName(appName)
//				.master("local")
                .getOrCreate();

        Dataset<Row> data = createDatasetFromCSV(spark, inputDir + "/kdd.data");

        System.out.println("Data:");
        data.show();


        List<Double> trainAccuracies = new ArrayList<>();
        List<Double> testAccuracies = new ArrayList<>();
        List<Long> runningTimes = new ArrayList<>();

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            Instant startTime = Instant.now();
            int SEED = (i+1) * 100;

            System.out.format("Iteration: %d - Seed: %d\n", i, SEED);

            Dataset<Row>[] splits = data.randomSplit(new double[]{0.7,0.3}, SEED);
            Dataset<Row> training = splits[0];
            Dataset<Row> test = splits[1];


            // Create model
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(100)			// Set maximum iterations (10)
                    .setRegParam(0.3)			// Set lambda (0.3)
                    .setElasticNetParam(0.8);	// Set Alpha (0.8)


            // Fit Model
            LogisticRegressionModel lrModel = lr.fit(training);

            // Model Information
            Vector coefficients = lrModel.coefficients();
            double intercept = lrModel.intercept();

            System.out.println("coefficients = " + coefficients);
            System.out.println("intercept = " + intercept);

            // Model training summary information
            BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();

            Dataset<Row> roc = trainingSummary.roc();
            double trainingAccuracy = trainingSummary.accuracy();
            double weightedPrecision = trainingSummary.weightedPrecision();
            double areaUnderROC = trainingSummary.areaUnderROC();

//            System.out.println("weightedPrecision = " + weightedPrecision);
//            System.out.println("areaUnderROC = " + areaUnderROC);


            // Report loss per iteration of the training
            double[] objectiveHistory = trainingSummary.objectiveHistory();
//            for (double lossPerIteration : objectiveHistory) {
//                System.out.println(lossPerIteration);
//            }

            System.out.format("[SEED: %d] - Training Accuracy: %f\n", SEED, trainingAccuracy);
            trainAccuracies.add(trainingAccuracy);


            // Get threshold of max f-measure
            Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
            double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
            double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                    .select("threshold")
                    .head()
                    .getDouble(0);
            lrModel.setThreshold(bestThreshold);


            // Make predictions
            Dataset<Row> predictions = lrModel.transform(test);
//            predictions.show(5);

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            double testAccuracy = evaluator.evaluate(predictions);
            testAccuracies.add(testAccuracy);

            Instant endTime = Instant.now();
            long timeElapsed = Duration.between(startTime, endTime).toMillis();
            runningTimes.add(timeElapsed);
        }


        // TRAINING ACCURACY
        double averageTrainAccuracy = trainAccuracies.stream()
                                              .mapToDouble(i -> i)
                                              .average()
                                              .orElse(0);
        double minTrainAccuracy = trainAccuracies.stream().mapToDouble(i -> i).min().orElse(0);
        double maxTrainAccuracy = trainAccuracies.stream().mapToDouble(i -> i).max().orElse(0);
        double trainingStandardDeviation = calculateSD(trainAccuracies);


        // TESTING ACCURACY
        double averageTestAccuracy = testAccuracies.stream()
                                            .mapToDouble(i -> i)
                                            .average()
                                            .orElse(0);
        double minTestAccuracy = testAccuracies.stream().mapToDouble(i -> i).min().orElse(0);
        double maxTestAccuracy = testAccuracies.stream().mapToDouble(i -> i).max().orElse(0);
        double testingStandardDeviation = calculateSD(testAccuracies);

        // TIMING
        double minRunningTime = runningTimes.stream().mapToLong(i -> i).min().orElse(0);
        double maxRunningTime = runningTimes.stream().mapToLong(i -> i).max().orElse(0);
        double averageRunningTime = runningTimes.stream().mapToLong(i -> i).average().orElse(0);



        System.out.println("\n\n");
        System.out.println("TRAINING:");
        System.out.println("Average: " + averageTrainAccuracy);
        System.out.println("Min: " + minTrainAccuracy);
        System.out.println("Max: " + maxTrainAccuracy);
        System.out.println("Standard Deviation: " + trainingStandardDeviation);
//        System.out.println("All Accuracies: " + trainAccuracies);


        System.out.println("\n");
        System.out.println("TESTING:");
        System.out.println("Average: " + averageTestAccuracy);
        System.out.println("Min: " + minTestAccuracy);
        System.out.println("Max: " + maxTestAccuracy);
        System.out.println("Standard Deviation: " + testingStandardDeviation);
//        System.out.println("All Accuracies: " + testAccuracies);


        System.out.println("\n");
        System.out.println("TIMING:");
        System.out.println("Average: " + averageRunningTime);
        System.out.println("Min: " + minRunningTime);
        System.out.println("Max: " + maxRunningTime);











        /*
        Report for training and test accuracy:

        max,
        min,
        average accuracy
        standard deviation

        obtained from the 10 runs
         */

//        System.out.println();

//        deleteDirectory(new File(outputDir));
//        predictions.toJavaRDD().saveAsTextFile(outputDir);

//        predictions.write().mode(SaveMode.Overwrite).json(outputDir);



    }
}
