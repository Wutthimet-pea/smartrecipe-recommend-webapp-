from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import *
from pyspark.ml.functions import vector_to_array

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
    

    col_reviews = ["recipeid","authorId","rating","review","datesubmitted"]
    col_recipes = ['recipeid','name','totaltime','reviewcount','calories','fatcontent','sodiumcontent','carbohydratecontent','proteincontent','recipeinstructions']

    df_reviews = df_reviews.select(*col_reviews)
    df_reciped = df_reciped.select(*col_recipes)
    
    return df_reciped, df_reviews

def pre_process_data(df_reciped, df_reviews):


    df_reciped=df_reciped.withColumn('instructionstep', F.size('recipeinstructions'))\
                         .drop('recipeinstructions')

    df_reciped = df_reciped.withColumn("hours", F.regexp_extract("totaltime", r"(\d+)H", 1).cast("int"))
    df_reciped = df_reciped.withColumn("minutes", F.regexp_extract("totaltime", r"(\d+)M", 1).cast("int"))
    df_reciped = df_reciped.fillna({"hours": 0, "minutes": 0})

    df_reciped = df_reciped.withColumn("totaltimeminutes", F.col("hours") * 60 + F.col("minutes"))
    df_reciped = df_reciped.drop(*["hours", "minutes","totaltime"])\
                           .withColumnRenamed("totaltimeminutes","time").fillna(0)
    

    df_reviews = df_reviews.withColumn("reviewlength", F.length("review")).drop("review")

    df_merge = df_reviews.join(df_reciped, on=["recipeid"], how="left")
    df_merge = df_merge.filter(F.col('Time').isNotNull())
    
    print("Join(Combine and drop null) : ",df_merge.count())

    agg_df = df_merge.groupBy("AuthorId").agg(
        F.avg("rating").alias("avg_rating"),
        F.avg("reviewLength").alias("avg_reviewlength"),
        F.count("*").alias("review_count"),
        F.avg("instructionstep").alias("avg_instructionlength"),
        F.avg("time").alias("avg_time"),
        F.avg("calories").alias("avg_calories"),
        F.avg("fatcontent").alias("avg_fatcontent"),
        F.avg("sodiumcontent").alias("avg_sodiumcontent"),
        F.avg("carbohydratecontent").alias("avg_carbohydratecontent"),
        F.avg("proteincontent").alias("avg_proteincontent"),
        ).drop('datesubmitted')
    
    print("Agg(Amount of Author) : ",agg_df.count())

    return df_merge, agg_df, df_reciped

def feature_engineering_and_vectorization(df_merge, agg_df, df_reciped):

    recipe_assembler = VectorAssembler(
        inputCols=[
            "reviewcount","calories","fatcontent","sodiumcontent","carbohydratecontent","proteincontent","instructionstep","time"
        ],
        outputCol="features"
    )
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledfeatures",
        withMean=True,     # subtract mean
        withStd=True       # divide by std dev
    )
    normalizer = Normalizer(
        inputCol="scaledfeatures", 
        outputCol="normfeatures", 
        p=2
    )
    join_vec_df = recipe_assembler.transform(df_merge)

    join_vec_df = scaler.fit(join_vec_df).transform(join_vec_df)

    join_vec_df = normalizer.transform(join_vec_df)

    recipe_vec_df = recipe_assembler.transform(df_reciped)

    final_recipe_vec_df = scaler.fit(recipe_vec_df).transform(recipe_vec_df)

    final_recipe_vec_df = normalizer.transform(final_recipe_vec_df)

    agg_assembler = VectorAssembler(
        inputCols=[
            "avg_rating", "avg_reviewlength", 
            "review_count", "avg_instructionlength", 
            "avg_time","avg_calories","avg_fatcontent","avg_sodiumcontent","avg_carbohydratecontent","avg_proteincontent"
        ],
        outputCol="features"
    )
    
    user_vec_df = agg_assembler.transform(agg_df)

    final_user_vec_df = scaler.fit(user_vec_df).transform(user_vec_df)

    final_user_vec_df = normalizer.transform(final_user_vec_df)

    return join_vec_df, final_recipe_vec_df

def get_sample_with_features(join_vec_df, final_recipe_vec_df, sample_size=20):
    # Step 1: Count recipes per author
    author_counts = join_vec_df.groupBy("authorid").agg(F.count("*").alias("cnt"))

    # Step 2: Filter for authors with >2 recipes
    frequent_authors = author_counts.filter(F.col("cnt") > 2).select("authorid")

    # Step 3: Keep only those authors
    filtered = join_vec_df.join(frequent_authors, on=["authorid"], how="inner")

    # Step 4: Remove the latest submission per author
    w_desc = Window.partitionBy("authorid").orderBy(F.col("datesubmitted").desc())
    filtered = filtered.withColumn("rn", F.row_number().over(w_desc)).filter(F.col("rn") > 1).drop("rn")

    # Step 5: Add NextRecipeId (ascending by DateSubmitted)
    w_asc = Window.partitionBy("authorid").orderBy("datesubmitted")
    filtered = filtered.withColumn("nextrecipeid", F.lead("recipeid").over(w_asc))

    # Step 6: Random sample
    sampled = filtered.orderBy(F.rand()).limit(sample_size)

    # Step 7: Join with vector & metadata
    # sampled = sampled.join(
    #     final_recipe_vec_df.select("RecipeId", "normFeatures", "Name"),
    #     on="RecipeId",
    #     how="left"
    # )
    sampled.drop("name")

    return sampled

def generate_cosine_similarity_results(rand10_df, final_vec_df, spark):
    results = []

    # Collect to local
    rand10_local = rand10_df.select(
        "authorid", "recipeid", "nextrecipeid", "normfeatures", "name"
    ).collect()

    for row in rand10_local:
        author_id = row["authorid"]
        recipe_id = row["recipeid"]
        next_recipe_id = row["nextrecipeid"]
        vec = row["normfeatures"]
        name = row["name"]

        # Get expected recipe name
        expect_name_row = final_vec_df.filter(F.col("recipeid") == next_recipe_id).select("name").limit(1).collect()
        expect_recipe_name = expect_name_row[0]["name"] if expect_name_row else None

        # Convert vector to array
        vec_arr = [float(x) for x in vec.toArray()]
        vec_lit = F.array([F.lit(x) for x in vec_arr])

        # Cosine similarity calculation
        similarity_df = final_vec_df \
            .filter(F.col("recipeid") != recipe_id) \
            .withColumn("cosine_sim", F.aggregate(
                F.zip_with(vector_to_array(F.col("normfeatures")), vec_lit, lambda x, y: x * y),
                F.lit(0.0),
                lambda acc, v: acc + v
            )) \
            .select("recipeid", "cosine_sim", "name") \
            .orderBy(F.desc("cosine_sim")) \
            .limit(1000)

        # Get top 3 suggestions
        top_3 = similarity_df.limit(3).collect()
        top_ids = [r["recipeid"] for r in top_3] + [None] * (3 - len(top_3))
        top_names = [r["name"] for r in top_3] + [None] * (3 - len(top_3))

        # Score
        top_sim_ids = [r["recipeid"] for r in similarity_df.select("recipeid").limit(1000).collect()]
        score = (1000 - top_sim_ids.index(next_recipe_id)) / 1000 if next_recipe_id in top_sim_ids else 0

        results.append(Row(
            reviewerid=str(author_id),
            recipeid=str(recipe_id),
            recipe_name=str(name),
            firstsuggestid=str(top_ids[0]),
            firstsuggestname=str(top_names[0]),
            secondsuggestid=str(top_ids[1]),
            secondsuggestname=str(top_names[1]),
            thirdsuggestid=str(top_ids[2]),
            thirdsuggestname=str(top_names[2]),
            score=float(score),
            expect_id=str(next_recipe_id),
            expect_recipe=str(expect_recipe_name),
        ))
    
    schema = T.StructType(
        [
            T.StructField("reviewerid", T.StringType(), True),
            T.StructField("recipeid", T.StringType(), True),
            T.StructField("recipe_name", T.StringType(), True),
            T.StructField("firstsuggestid", T.StringType(), True),
            T.StructField("firstsuggestname", T.StringType(), True),
            T.StructField("secondsuggestid", T.StringType(), True),
            T.StructField("secondsuggestname", T.StringType(),True),
            T.StructField("thirdsuggestid", T.StringType(), True),
            T.StructField("thirdsuggestname", T.StringType(), True),
            T.StructField("score", T.DoubleType(), True),
            T.StructField("expect_id", T.StringType(), True),
            T.StructField("expect_recipe", T.StringType(), True)
        ]
    )
       
    df_spark = spark.createDataFrame(results, schema)
    selected_columns = [
    "reviewerid", "recipeid", "recipe_name",
    "firstsuggestid", "firstsuggestname",
    "secondsuggestid", "secondsuggestname",
    "thirdsuggestid", "thirdsuggestname",
    "score", "expect_id", "expect_recipe"
    ]
    df_spark_selected = df_spark.select(*selected_columns)

    return df_spark_selected


def main():

    spark = SparkSession \
            .builder \
            .appName("Recommendation") \
            .config("spark.hadoop.io.native.lib", "false") \
            .getOrCreate()
    
    df_reciped, df_reviews = load_data(spark)
    
    print("***********join data**************")
    df_merge, agg_df, df_reciped = pre_process_data(df_reciped, df_reviews)

    print("***********turn to vector**************")
    join_vec_df, final_recipe_vec_df = feature_engineering_and_vectorization(df_merge, agg_df, df_reciped)

    sample = get_sample_with_features(join_vec_df, final_recipe_vec_df, sample_size=10)
    
    time_start_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"start training at: {time_start_train}")

    df_spark_selected = generate_cosine_similarity_results(sample, final_recipe_vec_df, spark)

    time_finish_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"finish training at: {time_finish_train}")
    training_duration = (time_finish_train - time_start_train).total_seconds()
    print(f"Training took: {training_duration:.2f} seconds")

    print("write file")
    time_start_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"start write file at: {time_start_train}")

    df_spark_selected.write.mode("overwrite").format("parquet").save(f"{PATH_MODEL}/cosine_similarity_result.parquet")

    time_finish_train = datetime.now(ZoneInfo("Asia/Bangkok"))
    print(f"finish write file at: {time_finish_train}")
    training_duration = (time_finish_train - time_start_train).total_seconds()
    print(f"Writing took: {training_duration:.2f} seconds")


    spark.stop()

if __name__ == "__main__":
    main()



