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

        final String inputDir = args[0]; // Location of the data file
        final int NUM_ITERATIONS = 10;

        SparkSession spark = SparkSession
                .builder()
                .appName("LogisticRegression")
//				.master("local") //! Only for local dev
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> data = createDatasetFromCSV(spark, inputDir + "/kdd.data");
        System.out.println("Input Data:");
        data.show();

        List<Double> trainAccuracies = new ArrayList<>();
        List<Double> testAccuracies = new ArrayList<>();
        List<Long> runningTimes = new ArrayList<>();

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            Instant startTime = Instant.now();
            int SEED = (i+1) * 100;

            System.out.format("\nSeed: [%d] (iteration %d)\n", SEED, i+1);

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
//            double[] objectiveHistory = trainingSummary.objectiveHistory();
//            for (double lossPerIteration : objectiveHistory) {
//                System.out.println(lossPerIteration);
//            }

            System.out.format("Training Accuracy: %f\n", trainingAccuracy);
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
            System.out.format("Test Accuracy: %f\n", testAccuracy);
            testAccuracies.add(testAccuracy);

            Instant endTime = Instant.now();
            long timeElapsed = Duration.between(startTime, endTime).toMillis();
            runningTimes.add(timeElapsed);
        }

        // Calculates and prints all results
        printAllResults(trainAccuracies, testAccuracies, runningTimes);

    }
}
