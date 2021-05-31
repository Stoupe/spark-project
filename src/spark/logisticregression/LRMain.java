package spark.logisticregression;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.io.File;

import static spark.HelperFunctions.deleteDirectory;


public class LRMain {



    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        final String appName = "SparkLogisticRegression";
        final String inputDir = args[0];
        final String outputDir = args[1];
        final int SEED = 123;

        SparkSession spark = SparkSession
                .builder()
                .appName(appName)
				.master("local")
                .getOrCreate();


        JavaRDD<String> lines = spark.sparkContext().textFile(inputDir + "/kdd.data",0).toJavaRDD();

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

        Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);

        data.show();
        System.out.println();


        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7,0.3}, SEED);

        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];


        // Create model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)				// Set maximum iterations
                .setRegParam(0.3)			// Set lambda
                .setElasticNetParam(0.8);	// Set Alpha

        // Fit model
        System.out.println("FITTING MODEL...");
        LogisticRegressionModel lrModel = lr.fit(training);

        System.out.println("Coefficients: " + lrModel.coefficients() + "\nIntercept: " + lrModel.intercept());


        // ==============================================================

        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();

        double[] objectiveHistory = trainingSummary.objectiveHistory();

        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }

        // Obtain the ROC as a dataframe and areaUnderROC
        Dataset<Row> roc = trainingSummary.roc();
        roc.show();
        roc.select("FPR").show();
        System.out.println(trainingSummary.areaUnderROC());



        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();

        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);

        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold")
                .head()
                .getDouble(0);

        lrModel.setThreshold(bestThreshold);

        Dataset<Row> predictions = lrModel.transform(test);

        predictions.show(5);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);

        System.out.println("Test Error = " + (1.0 - accuracy));


        /*
        Report for training and test accuracy:

        max,
        min,
        average accuracy
        standard deviation

        obtained from the 10 runs
         */



        deleteDirectory(new File(outputDir));
        roc.toJavaRDD().saveAsTextFile(outputDir);

    }
}
