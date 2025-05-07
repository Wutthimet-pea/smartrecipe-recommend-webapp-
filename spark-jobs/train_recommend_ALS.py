from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import pyspark.sql.types as T
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import json
from datetime import datetime
from zoneinfo import ZoneInfo

PATH_PROGRAM_LOCAL = "/home/g6780381426/recommendation_pyspark/"

with open(f'{PATH_PROGRAM_LOCAL}cfg/config_path.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

PATH_PROGRAM = config['program']
PATH_APP = f"{PATH_PROGRAM}{config['app']}"
PATH_MODEL = f"{PATH_PROGRAM}{config['model']}"
PATH_DATA = f"{PATH_PROGRAM}{config['data']}"


def load_data(spark):

    df_reciped = spark.read.parquet(f"{PATH_DATA}/recipes.parquet")
    print(f"df_reciped:\nrow: {df_reciped.count()}\ncol:{len(df_reciped.columns)}\n")
    df_reviews = spark.read.parquet(f"{PATH_DATA}/reviews.parquet")
    print(f"df_reviews:\nrow: {df_reviews.count()}\ncol:{len(df_reviews.columns)}")

    df_reciped = df_reciped.withColumn("RecipeId", F.col("RecipeId").cast(T.IntegerType()))

    for col in df_reviews.columns:
        df_reviews = df_reviews.withColumnRenamed(col, col.lower())
    for col in df_reciped.columns:
        df_reciped = df_reciped.withColumnRenamed(col, col.lower())
    
    col_reviews = ["recipeid", "authorid", "rating"]
    col_recipes = ["recipeid", "name", "recipeCategory"]

    df_reviews = df_reviews.select(*col_reviews)
    df_reciped = df_reciped.select(*col_recipes)
    
    return df_reciped, df_reviews

def eda(df_reciped, df_reviews):

    print("********************")
    print("********************")
    print("Reviews Data")
    print("********************")
    print("check the data types")
    df_reviews.printSchema()
    print("\nlook at some statistics")
    df_reviews.describe().show()
    print("\nnumber of ratings" , df_reviews.select('Rating').count())
    print("\nnumber of rated recipe", df_reviews.select('RecipeId').distinct().count())
    print("\nnumber of users", df_reviews.select('AuthorId').distinct().count())

    print("\nRating")
    df_reviews.groupBy("Rating").count().orderBy("Rating").show(truncate = False)
    print("\nAuthorId")
    df_reviews.groupBy("AuthorId").count().orderBy(F.col("count").desc()).show(truncate = False)
    print("\nRecipeId")
    df_reviews.groupBy("RecipeId").count().orderBy(F.col("count").desc()).show(truncate = False)
    print("Null check")
    df_reviews.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df_reviews.columns]).show()


    print("\n********************")
    print("********************")
    print("Reciped Data")
    print("********************")
    print("check the data types")
    df_reciped.printSchema()

    print("\nRecipeCategory")
    df_reciped.groupBy("RecipeCategory").count().orderBy(F.col("count").desc()).show(truncate = False, n = 5)
    print("\nName")
    df_reciped.groupBy("Name").count().orderBy(F.col("count").desc()).show(truncate = False, n = 5)

    print("\n********************")
    print("********************")
    print("Reciped X Reviews Data")
    print("********************")
    df_full = df_reviews.join(df_reciped, on = ["RecipeId"], how = "left")

    print("\nTop Recipe Category Reviewed")
    df_full.groupBy("RecipeCategory").count().orderBy(F.col("count").desc()).show(truncate = False, n = 5)
    print("\nTop Recipe Name Reviewed")
    df_full.groupBy("Name").count().orderBy(F.col("count").desc()).show(truncate = False, n = 5)
    print("\nTop Recipe Category Rating")
    df_full.groupBy("RecipeCategory").agg(F.mean(F.col("Rating")).alias("average_rating_by_cat")).orderBy(F.col("average_rating_by_cat").desc()).show(truncate = False, n = 5)
    print("\nTop Recipe Name Rating")
    df_full.groupBy("Name").agg(F.mean(F.col("Rating")).alias("average_rating_by_cat")).orderBy(F.col("average_rating_by_cat").desc()).show(truncate = False, n = 5)


def preprocess(df_reciped, df_reviews):

    train, test = df_reviews.randomSplit([0.90,0.10], 42)
    print(f"num training data: {train.count()}")
    print(f"num test data: {test.count()}")

    return train, test


def training_model(train_data):

    print("**********")
    print("Trainig Based Line Model")

    als = ALS(rank=10,
          userCol="authorid",
          itemCol="recipeid",
          # coldStartStrategy="drop",  # 🔥 Important line!
          ratingCol="rating",
        implicitPrefs=False)


    # Train ALS model on training data
    time_start_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"start training at: {time_start_train}")

    model_artifact = als.fit(train_data)

    print("DONE")
    time_finish_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"finish training at: {time_finish_train}")
    training_duration = (time_finish_train - time_start_train).total_seconds()
    print(f"Training took: {training_duration:.2f} seconds")

    print("**********")


    return model_artifact

def model_performance(test_data, model_artifact):

    print("**********************evaluate model**********************")
    predictions = model_artifact.transform(test_data).na.drop()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE: ", rmse)
    print("**********************evaluate model**********************")


def tuning_evaluate_model(train_data):

    als = ALS(userCol="authorid", itemCol="recipeid", ratingCol="rating", coldStartStrategy="drop", seed=42)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    paramGrid = ParamGridBuilder() \
        .addGrid(als.rank, [1, 5, 10, 15, 20, 25, 50, 75, 100]) \
        .addGrid(als.regParam, [0.000000001, 0.0000001, 0.00001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 15, 25, 50, 100]) \
        .addGrid(als.maxIter, [5, 10, 15, 25, 50, 100, 150]) \
        .build()
    
    validator = TrainValidationSplit(seed=42,
                                    estimator=als, # the model we wish to evaluate
                                    evaluator=evaluator, # the metric we wish to optimize against
                                    estimatorParamMaps=paramGrid, # list of Param Maps we created earlier
                                    trainRatio=0.9,
                                    )
    
    time_start_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"start tuning at: {time_start_train}")

    validator_model = validator.fit(train_data)

    print("DONE")
    time_finish_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"finish tuning at: {time_finish_train}")
    training_duration = (time_finish_train - time_start_train).total_seconds()
    print(f"Training took: {training_duration:.2f} seconds")

    best_model = validator_model.bestModel
    print("Best model parammap")
    print(best_model.extractParamMap())

    metrics = validator_model.validationMetrics
    params = validator_model.getEstimatorParamMaps()
    metrics_and_params = list(zip(metrics, params))
    metrics_and_params.sort(key=lambda x: x[0], reverse=True)

    return best_model


def main():

    spark = SparkSession \
            .builder \
            .appName("Recommendation") \
            .config("spark.hadoop.io.native.lib", "false") \
            .getOrCreate()
    
    df_reciped, df_reviews = load_data(spark)
    eda(df_reciped, df_reviews)

    train_data, test_data = preprocess(df_reciped, df_reviews)

    based_model_artifact = training_model(train_data)
    print("Test based line model")
    model_performance_based_line = model_performance(test_data, based_model_artifact)

    tune_model_artifact = tuning_evaluate_model(train_data)
    print("Test tuning model")
    model_performance_tune = model_performance(test_data, tune_model_artifact)

    model_name = f'alsmodel_{str(datetime.now().time()).replace(":", "").replace(".", "")}'

    print(f"save model to: {PATH_MODEL}/{model_name}")
    tune_model_artifact.mode("overwrite").save(f"{PATH_MODEL}/{model_name}")
    print("save Done")







    spark.stop()

if __name__ == "__main__":
    main()