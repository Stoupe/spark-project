package spark.decisiontree;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.FileWriter;
import java.io.IOException;


public class DTMain {


    public static void main(String[] args){

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        String appName = "SparkWordCount";

        SparkSession spark = SparkSession.builder()
                .appName(appName)
//                .master("local")
                .getOrCreate();

        JavaRDD<String> lines = spark.sparkContext().textFile("input/kdd.data",0)
                .toJavaRDD();

        JavaRDD<LabeledPoint> linesRDD = lines.map(line -> {
            String[]tokens = line.split(",");
            double[]features = new double[tokens.length -1];
            for (int i = 0; i < features.length; i++){
                features[i] = Double.parseDouble(tokens[i]);}
            Vector v = new DenseVector(features);
            if(tokens[features.length].equals("anomaly")){
                return new LabeledPoint(0.0, v);
            } else {return new LabeledPoint(1.0, v); }
        });

        Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);

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
        try {
            String outputFile = "output.txt";
            FileWriter myWriter = new FileWriter(outputFile);

            myWriter.write("Files in Java might be tricky, but it is fun enough!");
            myWriter.close();

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

//        predictions.toJavaRDD().saveAsTextFile("output");


    }
}

