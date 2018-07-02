import time
import pyspark
import os
import csv
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

sc = pyspark.SparkContext()
spark = SparkSession(sc)

politics_train_path='/home/hadoop/spark/politics/'
politics_train_rdd = sc.textFile(politics_train_path, use_unicode=False).map(lambda x: x.decode("utf-8"))

from pyspark.sql.types import StringType
df_politics= spark.createDataFrame(politics_train_rdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")

from pyspark.sql.functions import lit, col
df_politics = df_politics.withColumn('label',lit(0))

sports_train_path='/home/hadoop/spark/sports/'
sports_train_rdd = sc.textFile(sports_train_path, use_unicode=False).map(lambda x: x.decode("utf-8"))

df_sports= spark.createDataFrame(sports_train_rdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")

#print df_sports.count()

df_sports = df_sports.withColumn('label',lit(1))

business_train_path='/home/hadoop/spark/business/'
business_train_rdd = sc.textFile(business_train_path, use_unicode=False).map(lambda x: x.decode("utf-8"))

df_business= spark.createDataFrame(business_train_rdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")

#print df_business.count()

df_business = df_business.withColumn('label',lit(2))

media_train_path='/home/hadoop/spark/media/'
media_train_rdd = sc.textFile(media_train_path, use_unicode=False).map(lambda x: x.decode("utf-8"))

df_media= spark.createDataFrame(media_train_rdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")

df_media = df_media.withColumn('label',lit(3))

df_total1 = df_politics.union(df_sports)
df_total2 = df_business.union(df_media)
df_total = df_total1.union(df_total2)

from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import StopWordsRemover
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

tokenized = tokenizer.transform(df_total)

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="filtered")

from pyspark.ml.feature import HashingTF, IDF

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=200)
#df_Data = hashingTF.transform(df_clean)

idf = IDF(inputCol="rawFeatures", outputCol="features")

pipeline1 = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
model_politics = pipeline1.fit(df_total)
df_data2 = model_politics.transform(df_total)
df_data2.show(3)

rawTrainData, rawTestData = df_data2.randomSplit([0.8, 0.2])
rawTrainData.show(3)

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(rawTrainData)

# select example rows to display.
predictions = model.transform(rawTestData)
#predictions.show(3)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

#unknown_accuracy
politics_tp='/home/hadoop/spark/unknown_test/politics/'
politics_trdd = sc.textFile(politics_tp, use_unicode=False).map(lambda x: x.decode("utf-8"))
df1= spark.createDataFrame(politics_trdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")
df1 = df1.withColumn('label',lit(0))

business_tp='/home/hadoop/spark/unknown_test/business/'
business_trdd = sc.textFile(business_tp, use_unicode=False).map(lambda x: x.decode("utf-8"))
df2= spark.createDataFrame(business_trdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")
df2 = df2.withColumn('label',lit(2))

sports_tp='/home/hadoop/spark/unknown_test/sports/'
sports_trdd = sc.textFile(sports_tp, use_unicode=False).map(lambda x: x.decode("utf-8"))
df3= spark.createDataFrame(sports_trdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")
df3 = df3.withColumn('label',lit(1))

media_tp='/home/hadoop/spark/unknown_test/media/'
media_trdd = sc.textFile(media_tp, use_unicode=False).map(lambda x: x.decode("utf-8"))
df4= spark.createDataFrame(media_trdd.map(lambda x: x.encode("utf-8")), StringType()).toDF("sentence")
df4 = df4.withColumn('label',lit(3))

df5 = df1.union(df2)
df6 = df3.union(df4)
df7 = df5.union(df6)

model_u = pipeline1.fit(df7)
df_8 = model_u.transform(df7)

df_8.show(3)

predictions2 = model.transform(df_8)
#predictions2.show(3)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions2)
print("Unknown Test set accuracy = " + str(accuracy))
