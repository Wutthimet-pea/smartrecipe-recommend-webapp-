from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import pyspark.sql.types as T
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import datetime
import json
import datetime
from zoneinfo import ZoneInfo

PATH_PROGRAM_LOCAL = "/home/g6780381426/recommendation_pyspark/"

with open(f'{PATH_PROGRAM_LOCAL}cfg/config_path.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

PATH_PROGRAM = config['program']
PATH_APP = f"{PATH_PROGRAM}{config['app']}"
PATH_MODEL = f"{PATH_PROGRAM}{config['model']}"
PATH_DATA = f"{PATH_PROGRAM}{config['data']}"


# This Function use for ingestion data to system
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
    col_recipes = ["recipeid", "name", "recipecategory"]

    df_reviews = df_reviews.select(*col_reviews)
    df_reciped = df_reciped.select(*col_recipes)
    
    return df_reciped, df_reviews

# This Function use for predicting recommend for each user
def get_recomend_recipeid_for_user(model, reviewer_id, recipe, num_recommend, spark):

    data = [(x,) for x in reviewer_id]
    df_user = spark.createDataFrame(data, ["authorid"])
    recommend_df = model.recommendForUserSubset(df_user, num_recommend)
    
    for i in range(num_recommend):
        recommend_df = recommend_df.withColumn(f"recipeid{i+1}", F.col("recommendations")[i]["recipeid"])
        recommend_df = recommend_df.join(recipe, on = [recommend_df[f"recipeid{i+1}"] == recipe["recipeid"]], how = "left").drop("recipeid", "recipeCategory")
        recommend_df = recommend_df.withColumnRenamed("name", f"name{i+1}")
        recommend_df = recommend_df.withColumn(f"rating{i+1}", F.col("recommendations")[i]["rating"])

    recommend_df = recommend_df.drop("recommendations")
    

    return recommend_df.show(truncate = False)

# This Function use for predicting recommend for each reciped
def get_recomend_user_for_each_reciped(model, recipe_id, recipe, num_recommend, spark):

    data = [(x,) for x in recipe_id]
    df_recipe = spark.createDataFrame(data, ["recipeid"])
    recommend_df = model.recommendForItemSubset(df_recipe, num_recommend)
    recommend_df = recommend_df.join(recipe, on = ["recipeid"], how = "left")

    for i in range(num_recommend):
        recommend_df = recommend_df.withColumn(f"authorid{i+1}", F.col("recommendations")[i]["authorid"])
        recommend_df = recommend_df.withColumnRenamed("recipeCategory", f"recipeCategory{i+1}")
        recommend_df = recommend_df.withColumn(f"rating{i+1}", F.col("recommendations")[i]["rating"])

    recommend_df = recommend_df.drop("recommendations")

    return recommend_df.show(truncate = False)


def main():

    spark = SparkSession \
            .builder \
            .appName("Recommendation") \
            .config("spark.hadoop.io.native.lib", "false") \
            .getOrCreate()
    

    df_reciped, df_reviews = load_data(spark)

    list_user = df_reviews.select("AuthorId").distinct().rdd.map(lambda x: x.AuthorId).collect()

    model_name = "alsmodel"
    best_model_use = ALSModel.load(f"{PATH_MODEL}/{model_name}")

    time_start_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print("Start Recommend at: ", time_start_train)
    df_recommend = get_recomend_recipeid_for_user(best_model_use, list_user, df_reciped, 5, spark)

    time_finish_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"Finish Recommend at: {time_finish_train}")
    training_duration = (time_finish_train - time_start_train).total_seconds()
    print(f"Recommend took: {training_duration:.2f} seconds")

    print("Sart Write File")
    df_recommend.write.mode("overwrite").format("parquet").save(f"{PATH_MODEL}/ALS_recommend_result")
    print("Finish Write File")
    spark.stop()

if __name__ == "__main__":
    main()




