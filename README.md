# Spark Project

## Important Notes

To run on a local machine, the SparkSession initializer must include the `.master("local")` line.

To run on the hadoop cluster, this line must be removed/commented out before generating the JAR.

## To generate a JAR file with IntelliJ

`Build -> Build Artifacts -> SparkProject -> Build`

This will generate a JAR file in `out/artifacts/SparkProject`

## Logistic Regression

### Command to run:

`spark-submit --class spark.logisticregression.LRMain --master yarn --deploy-mode cluster SparkProject.jar input output`

## Decision Tree

### Command to run:

`spark-submit --class spark.decisiontree.DTMain --master yarn --deploy-mode cluster SparkProject.jar input output`