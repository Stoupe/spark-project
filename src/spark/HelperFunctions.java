package spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

public class HelperFunctions {

    public static void disableLogging() {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
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

    /**
     * Recursively delete a directory and all files within
     *
     * thanks https://mkyong.com/java/how-to-delete-directory-in-java/
     * @param file the directory to delete
     */
    public static void deleteDirectory(File file) {

        File[] list = file.listFiles();
        if (list != null) {
            for (File temp : list) {
                //recursive delete
                System.out.println("Visit " + temp);
                deleteDirectory(temp);
            }
        }

//        if (file.delete()) {
//            System.out.printf("Delete : %s%n", file);
//        } else {
//            System.err.printf("Unable to delete file or directory : %s%n", file);
//        }

    }

}
