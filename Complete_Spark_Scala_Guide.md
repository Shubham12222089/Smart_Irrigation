# Complete Spark & Scala Preparation Guide

This guide provides a comprehensive overview of the concepts required for your ETP, with a focus on Scala as the programming language.

---

## 1. Introduction to Spark

### 1.1. Limitations of MapReduce in Hadoop

MapReduce is a powerful model for batch processing large datasets, but it has several limitations that Spark was designed to overcome:

-   **Slow Performance for Iterative Jobs**: MapReduce writes intermediate results to disk (HDFS). For iterative algorithms (like many machine learning algorithms) that need to access the same data multiple times, this disk I/O creates significant overhead and slows down processing.
-   **No Real-Time Processing**: MapReduce is strictly a batch-oriented system. It is not suitable for real-time or near-real-time stream processing.
-   **Complex Programming Model**: Writing MapReduce jobs, especially complex ones, requires a lot of boilerplate code and can be difficult to manage.
-   **Manual Optimization**: Developers are often responsible for manually optimizing their MapReduce jobs, which can be a challenging task.

### 1.2. Comparison of Batch vs. Real-Time Analytics

| Feature             | Batch Analytics                               | Real-Time Analytics (Stream Processing)       |
| :------------------ | :-------------------------------------------- | :-------------------------------------------- |
| **Data Scope**      | Processes large, static datasets (bounded data). | Processes continuous streams of data (unbounded data). |
| **Latency**         | High latency (minutes to hours).              | Low latency (milliseconds to seconds).        |
| **Throughput**      | High throughput, optimized for large volumes. | Can also have high throughput, but prioritizes speed. |
| **Use Case**        | ETL jobs, historical reporting, complex data analysis. | Fraud detection, real-time monitoring, live recommendations. |
| **Example Systems** | Hadoop MapReduce, Spark (in batch mode).      | Spark Streaming, Apache Flink, Apache Kafka.  |

### 1.3. Application of Stream Processing and In-Memory Processing

-   **Stream Processing**: Enables applications to react to data as it arrives.
    -   **Financial Services**: Real-time fraud detection in credit card transactions.
    -   **IoT (Internet of Things)**: Monitoring sensor data from industrial equipment to predict failures.
    -   **Social Media**: Analyzing trending topics and user sentiment in real-time.
-   **In-Memory Processing**: The core feature that gives Spark its speed. By loading data into the memory of a distributed cluster and keeping it there for subsequent operations, Spark avoids the slow disk I/O of MapReduce.
    -   **Iterative Machine Learning**: Algorithms like K-Means or Logistic Regression can run much faster by caching the training data in memory for each iteration.
    -   **Interactive Data Analysis**: Data scientists can interactively query a large dataset with low latency because the data resides in memory.

### 1.4. Features and Benefits of Spark

-   **Speed**: Up to 100x faster than Hadoop MapReduce for in-memory processing and 10x faster on disk.
-   **Ease of Use**: Offers rich APIs in Scala, Java, Python, and R, with less boilerplate code than MapReduce.
-   **Unified Engine**: Supports batch processing, SQL queries (Spark SQL), real-time streaming (Spark Streaming), machine learning (MLlib), and graph processing (GraphX) within a single framework.
-   **Lazy Evaluation**: Transformations are not executed until an action is triggered, allowing the Catalyst optimizer to create the most efficient execution plan.
-   **Fault Tolerance**: Achieved through Resilient Distributed Datasets (RDDs), which can be recomputed from their lineage if a node fails.

### 1.5. Installation of Spark as a Standalone User

A standalone cluster is the simplest way to run Spark. It includes a master process and several worker processes.

1.  **Prerequisites**: Install Java (version 8/11).
2.  **Download Spark**: Download a pre-built version of Spark from the [official Apache Spark website](https://spark.apache.org/downloads.html).
3.  **Unpack the Archive**: `tar -xzf spark-3.3.0-bin-hadoop3.tgz`
4.  **Start the Master**: Navigate to the Spark directory and run: `./sbin/start-master.sh`. This will start a master process on your machine and provide a URL (e.g., `spark://<your-host>:7077`).
5.  **Start a Worker**: Start one or more worker processes and connect them to the master: `./sbin/start-worker.sh spark://<your-host>:7077`.
6.  **Submit an Application**: Use `spark-submit` to run your application on the cluster, or connect to it using the `spark-shell`.

### 1.6. Compare Spark vs. Hadoop Ecosystem

| Aspect              | Hadoop Ecosystem                               | Apache Spark                                   |
| :------------------ | :--------------------------------------------- | :--------------------------------------------- |
| **Core Processing** | Hadoop MapReduce (batch processing).           | Spark Core Engine (in-memory, batch, streaming). |
| **Speed**           | Slower due to disk-based processing.           | Much faster due to in-memory processing.       |
| **Data Sources**    | Primarily HDFS.                                | HDFS, Cassandra, HBase, S3, local files, etc.  |
| **Real-Time**       | No. Requires other tools like Storm or Flink.  | Yes, via Spark Streaming.                      |
| **Ease of Use**     | Low-level, verbose MapReduce API.              | High-level, concise APIs.                      |
| **Ecosystem**       | A collection of separate tools (Hive, Pig, Mahout). | A single, unified framework (SQL, MLlib, GraphX). |

**Conclusion**: Spark is not a replacement for Hadoop, but an alternative to Hadoop MapReduce. Spark often runs on top of HDFS for storage and can be managed by YARN, making it a powerful component *within* the broader Hadoop ecosystem.

---

## 2. Introduction to Programming in Scala

### 2.1. Features of Scala

Scala is a modern, multi-paradigm programming language designed to be concise and expressive.

-   **Object-Oriented**: Every value is an object, and every operation is a method call.
-   **Functional**: Functions are first-class citizens. You can pass them as arguments, return them from other functions, and assign them to variables.
-   **Statically Typed**: The compiler checks for type errors at compile time, which helps catch bugs early.
-   **Type Inference**: You often don't need to specify types explicitly; the compiler can infer them for you.
-   **Concise Syntax**: Scala's syntax is less verbose than Java's.
-   **JVM Interoperability**: Scala runs on the Java Virtual Machine (JVM) and has seamless interoperability with Java code.

### 2.2. Basic Data Types and Literals

| Type      | Description                  | Example Literal |
| :-------- | :--------------------------- | :-------------- |
| `Byte`    | 8-bit signed integer         | `127`           |
| `Short`   | 16-bit signed integer        | `32767`         |
| `Int`     | 32-bit signed integer        | `100`           |
| `Long`    | 64-bit signed integer        | `100L`          |
| `Float`   | 32-bit floating-point number | `3.14f`         |
| `Double`  | 64-bit floating-point number | `3.14159`       |
| `Char`    | 16-bit Unicode character     | `'A'`           |
| `String`  | A sequence of characters     | `"Hello Scala"` |
| `Boolean` | True or false                | `true` or `false` |
| `Unit`    | Corresponds to `void` in Java. Represents no value. | `()` |

### 2.3. Operators and Methods

In Scala, operators are just methods. For example, `1 + 2` is syntactic sugar for `1.+(2)`.

```scala
val a = 10
val b = 20

// Arithmetic
println(s"Sum: ${a + b}")
println(s"Product: ${a * b}")

// Relational
println(s"Is a > b? ${a > b}")

// Logical
val isSunny = true
val isWarm = false
println(s"Go to the beach? ${isSunny && isWarm}")
```

### 2.4. Introduction to Type Inference

The Scala compiler can often figure out the type of a variable on its own.

```scala
val name = "Alice" // Compiler infers type String
val age = 30        // Compiler infers type Int
val pi = 3.14       // Compiler infers type Double
```

### 2.5. Mutable vs. Immutable Collections

-   **Immutable**: Cannot be changed after creation. When you "add" an element, a new collection is created. This is the default in Scala and is preferred for safety and predictability.
-   **Mutable**: Can be modified in place.

```scala
// Immutable List (default)
val immutableList = List(1, 2, 3)
val newList = immutableList :+ 4 // Creates a new list: List(1, 2, 3, 4)
// immutableList is still List(1, 2, 3)

// Mutable List (requires explicit import)
import scala.collection.mutable.ListBuffer
val mutableList = ListBuffer(1, 2, 3)
mutableList += 4 // Modifies the list in place: ListBuffer(1, 2, 3, 4)
```

### 2.6. Lists, Maps, and Streams in Scala

#### A. Lists
An immutable, finite sequence of elements.

```scala
val fruits = List("Apple", "Banana", "Cherry")
println(fruits.head)      // Apple
println(fruits.tail)      // List(Banana, Cherry)
println(fruits(1))        // Banana
fruits.foreach(println)   // Prints each fruit on a new line
```

#### B. Maps
A collection of key-value pairs. By default, maps are immutable.

```scala
val ages = Map("Alice" -> 30, "Bob" -> 25)
println(ages("Alice")) // 30

// Adding a new entry creates a new map
val newAges = ages + ("Charlie" -> 35)
println(newAges) // Map(Alice -> 30, Bob -> 25, Charlie -> 35)
```

#### C. Streams
A `Stream` is like a `List`, but its elements are computed lazily. This is useful for representing very long or infinite sequences.

```scala
val streamOfNumbers = Stream.from(1) // An infinite stream of integers starting from 1
println(streamOfNumbers.take(5).toList) // List(1, 2, 3, 4, 5)
// Only the first 5 elements are computed.
```

### 2.7. Functions in Scala

Functions are first-class citizens.

```scala
// A named function
def add(x: Int, y: Int): Int = {
  return x + y
}

// An anonymous function (or lambda)
val multiply = (x: Int, y: Int) => x * y

println(add(5, 3))       // 8
println(multiply(5, 3))  // 15

// Functions can be passed as arguments
val numbers = List(1, 2, 3)
val doubledNumbers = numbers.map(n => n * 2) // Pass an anonymous function to map
println(doubledNumbers) // List(2, 4, 6)
```

---

## 3. Using RDD for Creating Applications in Spark

*(This section provides Scala examples for the RDD concepts)*

### 3.1. Creating RDDs

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder.appName("RDDsInScala").master("local[*]").getOrCreate()
val sc = spark.sparkContext // Get the SparkContext

// 1. From a collection
val data = Array(1, 2, 3, 4, 5)
val rdd = sc.parallelize(data)

// 2. From a file
val textRdd = sc.textFile("path/to/your/file.txt")
```

### 3.2. RDD Operations and Methods

#### A. Transformations (Lazy)

```scala
val numbersRdd = sc.parallelize(1 to 10)

// map: multiply each element by 2
val mappedRdd = numbersRdd.map(x => x * 2)

// filter: keep only even numbers
val filteredRdd = numbersRdd.filter(x => x % 2 == 0)

// flatMap: split lines into words
val lines = sc.parallelize(Seq("hello world", "spark is fun"))
val wordsRdd = lines.flatMap(line => line.split(" "))

// reduceByKey: count word occurrences
val wordPairs = wordsRdd.map(word => (word, 1))
val wordCounts = wordPairs.reduceByKey((a, b) => a + b)
```

#### B. Actions (Trigger Execution)

```scala
// collect: bring all elements to the driver (use with caution!)
val allWords = wordCounts.collect()
allWords.foreach(println)

// count: get the number of elements
println(s"Total numbers: ${numbersRdd.count()}")

// take(n): get the first n elements
val firstThree = numbersRdd.take(3)
println(firstThree.mkString(", ")) // 1, 2, 3

// reduce: aggregate all elements
val sum = numbersRdd.reduce((a, b) => a + b)
println(s"Sum: $sum") // 55

// saveAsTextFile: save to a file
// wordCounts.saveAsTextFile("output/wordcounts")
```

### 3.3. Invoking the Spark Shell

The Scala Spark shell is an interactive environment for running Spark code.

```bash
# In your terminal, navigate to the Spark directory and run:
./bin/spark-shell

# The 'sc' (SparkContext) and 'spark' (SparkSession) variables are pre-configured.
scala> val rdd = sc.parallelize(1 to 100)
scala> rdd.count()
res0: Long = 100
```

### 3.4. Shared Variables

#### A. Broadcast Variables (Read-only)

```scala
val lookupTable = Map("US" -> "United States", "CA" -> "Canada")
val broadcastVar = sc.broadcast(lookupTable)

val countryCodes = sc.parallelize(Seq("US", "CA", "MX"))
val fullNames = countryCodes.map(code => broadcastVar.value.getOrElse(code, "Unknown"))

println(fullNames.collect().mkString(", ")) // United States, Canada, Unknown
```

#### B. Accumulators (Write-only, for counters/sums)

```scala
// Note: For Spark 2.0+, it's recommended to use LongAccumulator
val counter = sc.longAccumulator("MyCounter")

val dataRdd = sc.parallelize(1 to 10)
dataRdd.foreach(x => counter.add(1))

println(s"Counter value: ${counter.value}") // 10
```

---

## 4. Running SQL Queries Using Spark SQL

### 4.1. Converting RDDs to DataFrames

A DataFrame is a distributed collection of data organized into named columns.

```scala
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

// 1. Inferring the Schema (using case classes)
case class Person(name: String, age: Int)
val peopleRdd = sc.parallelize(Seq(Person("Alice", 30), Person("Bob", 25)))
val peopleDF = spark.createDataFrame(peopleRdd)
peopleDF.show()

// 2. Programmatically Specifying the Schema
val rowRdd = sc.parallelize(Seq(Row("Charlie", 35), Row("David", 40)))
val schema = StructType(Array(
  StructField("name", StringType, true),
  StructField("age", IntegerType, true)
))
val dfFromRows = spark.createDataFrame(rowRdd, schema)
dfFromRows.show()
```

### 4.2. Concepts of Spark SQL

You can run SQL queries by registering a DataFrame as a temporary view.

```scala
// Create a temporary view
peopleDF.createOrReplaceTempView("people")

// Run SQL queries
val teenagersDF = spark.sql("SELECT name, age FROM people WHERE age BETWEEN 13 AND 19")
teenagersDF.show()

// You can also use the DataFrame API
peopleDF.filter("age > 28").show()
peopleDF.groupBy("age").count().show()
```

### 4.3. Concept of Hive Integration

To use Hive, you must enable Hive support in your `SparkSession`.

```scala
import org.apache.spark.sql.SparkSession

// This enables Hive support and connects to the Hive metastore
val spark = SparkSession.builder()
  .appName("HiveIntegration")
  .enableHiveSupport() // Key method
  .getOrCreate()

// Now you can query Hive tables directly
spark.sql("USE my_hive_database")
spark.sql("SELECT * FROM my_hive_table").show()
```

---

## 5. Spark Streaming with Apache Kafka

### 5.1. Kafka Fundamentals & Architecture

(As described in the previous guide: Topic, Producer, Consumer, Broker, Partition, Cluster, Replication).

### 5.2. Integration of Apache Kafka with Spark

You need the `spark-sql-kafka-0-10` package. In `sbt` (the Scala Build Tool), you would add this to your `build.sbt` file:

```sbt
libraryDependencies += "org.apache.spark" %% "spark-sql-kafka-0-10" % "3.3.0"
```

### 5.3. Real-Time Spark Streaming with Apache Spark (Structured Streaming)

Structured Streaming is the modern, DataFrame-based API for streaming.

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder
  .appName("KafkaStreamingExample")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Create a streaming DataFrame that reads from a Kafka topic
val kafkaDF = spark.readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", "localhost:9092") // Your Kafka broker
  .option("subscribe", "test-topic")
  .load()

// Kafka messages are binary, so cast the value to a string
val lines = kafkaDF.selectExpr("CAST(value AS STRING)").as[String]

// Perform a word count
val wordCounts = lines.flatMap(_.split(" ")).groupBy("value").count()

// Write the output to the console
val query = wordCounts.writeStream
  .outputMode("complete") // 'complete' shows the full result table each time
  .format("console")
  .start()

query.awaitTermination()
```

### 5.4. Kafka Consumers in Group

A consumer group allows a set of consumer processes to work together to consume a topic. Each partition is assigned to exactly one consumer in the group, enabling parallel processing and load balancing. If a consumer fails, its partitions are reassigned to another consumer in the group.

### 5.5. Kafka Java Programming

Since Scala runs on the JVM, you can use the standard Kafka Java client libraries directly within your Scala code to create producers and consumers.

**Producer Example Snippet (in Scala using Java library):**

```scala
import java.util.Properties
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}

val props = new Properties()
props.put("bootstrap.servers", "localhost:9092")
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

val producer = new KafkaProducer[String, String](props)
val record = new ProducerRecord[String, String]("test-topic", "key", "hello from scala")
producer.send(record)
producer.close()
```

---

## 6. Spark ML Programming and Graph Analytics

### 6.1. Spark MLlib and Key Concepts

`spark.ml` is the primary, DataFrame-based API.
-   **Transformer**: An algorithm that transforms a DataFrame (e.g., `StringIndexer`).
-   **Estimator**: An algorithm that is trained on a DataFrame to produce a Transformer (e.g., `LogisticRegression`).
-   **Pipeline**: Chains multiple stages (Transformers and Estimators) into a single workflow.

### 6.2. Spark ML Algorithms (Feature Extraction, Transformation, Classification)

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Sample data
val data = spark.createDataFrame(Seq(
  (0, "a", 1.0, "cat1"),
  (1, "b", 2.0, "cat2"),
  (2, "c", 3.0, "cat1"),
  (3, "a", 4.0, "cat2")
)).toDF("id", "feature1_str", "feature2_num", "label")

// 1. Feature Transformation
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
val featureIndexer = new StringIndexer().setInputCol("feature1_str").setOutputCol("feature1_idx")

// 2. Feature Assembler
val assembler = new VectorAssembler()
  .setInputCols(Array("feature1_idx", "feature2_num"))
  .setOutputCol("features")

// 3. Algorithm
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("features")

// 4. Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, assembler, dt))

// Train the model
val model = pipeline.fit(data)

// Make predictions
val predictions = model.transform(data)
predictions.select("prediction", "indexedLabel", "features").show()

// Evaluate the model
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Accuracy = $accuracy")
```

### 6.3. Frequent Pattern Mining (FP-Growth)

```scala
import org.apache.spark.ml.fpm.FPGrowth

val dataset = spark.createDataFrame(Seq(
  (0, Array("a", "b", "c")),
  (1, Array("a", "b", "d")),
  (2, Array("a", "c"))
)).toDF("id", "items")

val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)
val model = fpgrowth.fit(dataset)

// Display frequent itemsets
model.freqItemsets.show()

// Display generated association rules
model.associationRules.show()
```

### 6.4. Introduction to Spark GraphX

GraphX is Spark's API for graph computation, built on RDDs.

-   **Graph Representation**: A graph is represented by an RDD of vertices (`RDD[(VertexId, V)]`) and an RDD of edges (`RDD[Edge[E]]`). `VertexId` is a `Long`.

### 6.5. Spark GraphX Features and Operations

```scala
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

// Create RDDs for vertices and edges
val users: RDD[(VertexId, (String, String))] =
  sc.parallelize(Array((3L, ("rxin", "student")), (7L, ("jgonzal", "postdoc")),
                       (5L, ("franklin", "prof")), (2L, ("istoica", "prof"))))

val relationships: RDD[Edge[String]] =
  sc.parallelize(Array(Edge(3L, 7L, "collab"), Edge(5L, 3L, "advisor"),
                       Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi")))

// Define a default user in case of missing vertices
val defaultUser = ("John Doe", "Missing")

// Build the graph
val graph = Graph(users, relationships, defaultUser)

// --- Graph Operations ---

// 1. Count vertices and edges
println(s"Total users: ${graph.numVertices}")
println(s"Total relationships: ${graph.numEdges}")

// 2. Triplets: View the graph as (srcAttr, edgeAttr, dstAttr)
graph.triplets.map(
  triplet => s"${triplet.srcAttr._1} is the ${triplet.attr} of ${triplet.dstAttr._1}"
).collect().foreach(println)

// 3. PageRank
val pageRankGraph = graph.pageRank(0.001)
pageRankGraph.vertices.join(users).sortBy(_._2._1, ascending=false).map {
  case (id, (pr, (name, role))) => s"$name has PageRank $pr"
}.collect().foreach(println)
```
