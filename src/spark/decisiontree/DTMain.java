package spark.decisiontree;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

import static spark.HelperFunctions.createDatasetFromCSV;
import static spark.HelperFunctions.deleteDirectory;


public class DTMain {


    public static void main(String[] args){

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        String appName = "SparkWordCount";
        final String inputDir = args[0];
        final String outputDir = args[1];
        final int SEED = 123;

        SparkSession spark = SparkSession.builder()
                .appName(appName)
//                .master("local")
                .getOrCreate();


        Dataset<Row> data = createDatasetFromCSV(spark, inputDir + "/kdd.data");


        data.show();

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7,0.3},123);
        Dataset<Row> trainingData =splits[0];
        Dataset<Row> testData = splits[1];

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt,
                        labelConverter});

        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        predictions.select("predictedLabel", "label", "features").show(10);

        MulticlassClassificationEvaluator evaluator = new
                MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);

        System.out.println("Test Error = " + (1.0 - accuracy));

//        DecisionTreeClassificationModel treeModel =
//                (DecisionTreeClassificationModel) (model.stages()[2]);
//
//        System.out.println("Learned classification tree model:\n" +
//                treeModel.toDebugString());


        /*
        Report:

        max,
        min,
        average accuracy
        standard deviation

        obtained from the 10 runs
         */
        deleteDirectory(new File(outputDir));
        predictions.toJavaRDD().saveAsTextFile(outputDir);

    }
}

