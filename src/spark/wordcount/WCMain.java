package spark.wordcount;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.Arrays;

public class WCMain {
    public static void main(String[] args) {
        String appName = "SparkWordCount";

        SparkSession spark = SparkSession.builder()
                .appName(appName)
                //.master("local")
                .getOrCreate();

        JavaRDD<String> lines = spark.read().textFile(args[0]).javaRDD();

        JavaPairRDD<String, Integer> records =
                lines
                        .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
                        .mapToPair(word -> new Tuple2<String, Integer>(word, 1))
                        .reduceByKey((x,y) -> x+y);

        //records = records.sortByKey(true);
        //JavaRDD<String> formatted = records.map( new  Function<Tuple2<String,Integer>, String>()
        //{
        //
        //	@Override
        //	public String call(Tuple2<String, Integer> theTuple) throws  Exception {
        //		return  theTuple._1() +  " appears "  + theTuple._2();
        //	}
        //});

        records.repartition(1).saveAsTextFile(args[1]);
    }
}




