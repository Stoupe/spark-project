package spark.decisiontree;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static spark.HelperFunctions.*;

public class DTMain {

    public static void main(String[] args){

        final String inputDir = args[0]; // Location of the data file
        int NUM_ITERATIONS = 10;

        SparkSession spark = SparkSession.builder()
                .appName("DecisionTree")
//                .master("local") //! Only for local dev
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
            int SEED = (i + 1) * 100;

            System.out.format("\nSeed: [%d] (iteration %d)\n", SEED, i+1);

            // Index Labels
            StringIndexerModel labelIndexer = new StringIndexer()
                    .setInputCol("label")
                    .setOutputCol("indexedLabel")
                    .fit(data);

            // Index features
            VectorIndexerModel featureIndexer = new VectorIndexer()
                    .setInputCol("features")
                    .setOutputCol("indexedFeatures")
                    .setMaxCategories(4)
                    .fit(data);

            // Split the data
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.7,0.3}, SEED);
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            // Train the decision tree model
            DecisionTreeClassifier dt = new DecisionTreeClassifier()
                    .setLabelCol("indexedLabel")
                    .setFeaturesCol("indexedFeatures")
                    .setMaxDepth(10); // by default this value is 5


            // Convert labels back to original labels
            IndexToString labelConverter = new IndexToString()
                    .setInputCol("prediction")
                    .setOutputCol("predictedLabel")
                    .setLabels(labelIndexer.labels());

            // Add stages to a pipeline
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[] {
                            labelIndexer,
                            featureIndexer,
                            dt,
                            labelConverter
                    });

            // Run the pipeline
            PipelineModel model = pipeline.fit(trainingData);

            // Make predictions with training data
            Dataset<Row> trainingPredictions = model.transform(trainingData);

            // Make predictions with test data
            Dataset<Row> predictions = model.transform(testData);

            // Display first 10 predictions
            //predictions.select("predictedLabel", "label", "features").show(10);

            // Configure evaluator
            MulticlassClassificationEvaluator evaluator = new
                    MulticlassClassificationEvaluator()
                    .setLabelCol("indexedLabel")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            // Calculate training accuracy
            double trainAccuracy = evaluator.evaluate(trainingPredictions);
            System.out.println("Training Accuracy: " + trainAccuracy);
            trainAccuracies.add(trainAccuracy);

            // Calculate test accuracy
            double testAccuracy = evaluator.evaluate(predictions);
            System.out.println("Test Accuracy: " + testAccuracy);
            testAccuracies.add(testAccuracy);

            // Generate tree from model
            DecisionTreeClassificationModel treeModel =
                    (DecisionTreeClassificationModel) (model.stages()[2]);

            System.out.println("Number of Features selected: " + treeModel.featureImportances().numNonzeros());
            System.out.println("Feature Importance: " + treeModel.featureImportances());

            // Print generated decision tree
//            System.out.println("Learned classification tree model:\n" +
//                    treeModel.toDebugString());

            Instant endTime = Instant.now();
            long timeElapsed = Duration.between(startTime, endTime).toMillis();
            runningTimes.add(timeElapsed);
        }

        // Calculates and prints all results
        printAllResults(trainAccuracies, testAccuracies, runningTimes);

    }
}

