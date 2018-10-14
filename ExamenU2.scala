import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val spark = SparkSession.builder.master("local[*]").getOrCreate()
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","class")

val n_col = when($"class".contains("Iris-setosa"), 1.0).otherwise(when($"class".contains("Iris-virginica"), 3.0).otherwise(2.0))
val n_df = df.withColumn("n_tagger", n_col)
n_df.select("n_tagger","SepalLength","SepalWidth","PetalLength","PetalWidth","class").show(150, false)

//Transformacion de datos
val features = assembler.transform(n_df)
val assembler = new VectorAssembler().setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth","n_tagger"))
.setOutputCol("features")features.show(5)

///Toma todos los datos de la columna indexada (label) que hay en el datset para incluirlos al index
val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

//Los valores que se indentifican son indexados y se añade maxCategories para que los valores mayores a 4 sean tratados como continuos
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

//Los valores son tomados de manera aleatoria.
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val n_Data = splits(1)

//Las capas de red neuronal estan constituidas de una entrada de 5, 2 intermediarios tamaño 5 y una salia tamaño 3
val layers = Array[Int](5, 5, 5, 3)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
.setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

//Los labels que se fueron indexados son restaurados a los label originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Procesa el flujo de trabajo , aprende las predicciones del modelo usando los vectores
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))
val model = pipeline.fit(trainingData)

val predictions = model.transform(n_Data)predictions.show(5)

//Selecciona los features y la prueba de error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
