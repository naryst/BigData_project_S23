"""Module provide all used function for project"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import expr
SPARK, SC = None, None


def set_connect():
    """
    Set spark connection
    :return: None
    """
    global SPARK, SC

    SPARK = SparkSession.builder \
        .appName("Anime Project") \
        .master("local[*]") \
        .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.avro.compression.codec", "snappy") \
        .config("spark.jars",
                "file:///usr/hdp/current/hive-client/lib/"
                "hive-metastore-1.2.1000.2.6.5.0-292.jar,"
                "file:///usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar") \
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.0.3") \
        .config('spark.ui.showConsoleProgress', 'false') \
        .enableHiveSupport() \
        .getOrCreate()

    SC = SPARK.sparkContext
    return


def close_connect():
    """
    Close spark connection
    :return: None
    """
    SC.stop()
    return


def read_tables():
    """
    Read tables
    :return: tables anime and anime_list
    """

    anime = SPARK.read.format("avro").table('projectdb.anime')
    anime.createOrReplaceTempView('anime')

    anime_list = SPARK.read.format("avro").table('projectdb.anime_list')
    anime_list.createOrReplaceTempView('anime_list')

    return anime, anime_list


def preprocess_data(anime, anime_list):
    """
    Preprocess data
    :param anime: anime table
    :param anime_list: anime_list table
    :return: preprocessed data
    """
    anime_names = anime.select("MAL_ID", "Name", "Popularity")\
        .withColumnRenamed("MAL_ID", "anime_id")

    # Add two new columns to the DataFrame:
    # 1. "total_votes" which counts the number of votes for each anime
    # 2. "user_total_votes" which sums up the number of votes by each user for each anime
    anime_list = anime_list.filter(anime_list.watch_status == 2).drop(anime_list.watched_episodes)
    anime_list = anime_list.join(anime_names, anime_list.anime_id == anime_names.anime_id, "inner")\
        .drop(anime_names.anime_id)

    # Filter the DataFrame to include only the data where
    # 1. the anime has more than 5000 votes, and
    # 2. the users have given more than 1500 votes to the anime
    anime_list = anime_list.withColumn("anime_total_votes", count("*")
                                       .over(Window.partitionBy("anime_id")))\
        .withColumn("user_total_votes", count("*")
                    .over(Window.partitionBy("user_id")))
    anime_df_filtered = anime_list\
        .filter((anime_list.anime_total_votes > 5000) & (anime_list.user_total_votes > 1500))
    return anime_df_filtered


def create_matrix(anime_df_filtered):
    """
    Create User-Item matrix
    :param anime_df_filtered:
    :return: table with new columns (user_id_index and anime_id_index)
    """
    indexer = [StringIndexer(inputCol=column, outputCol=column + "_index")
               for column in ["user_id", "anime_id"]]
    pipeline = Pipeline(stages=indexer)
    transformed = pipeline.fit(anime_df_filtered).transform(anime_df_filtered)

    return transformed


def baseline(train_df):
    """
    Baseline solution. Give to all users top-5 popular anime
    :param train_df: train table
    :return: prediction
    """
    def get_most_pop(text):
        """
        set top-5 anime
        :return: top-5 anime
        """
        return most_pop_bc.value

    # find top-5 anime
    temp_arr = train_df.dropDuplicates(["anime_id_index"])\
        .sort("Popularity").filter(train_df.Popularity > 0).limit(5).collect()

    most_pop = []
    for element in temp_arr:
        most_pop.append(element.anime_id_index)

    # add to each users top-5 anime
    most_pop_bc = SC.broadcast(most_pop)
    func_udf = udf(get_most_pop, ArrayType(DoubleType()))

    # make a prediction format
    baseline_pred = (
        train_df.withColumn("pred_list", func_udf(train_df.user_id_index))
        .dropDuplicates(["user_id_index", "pred_list"])
        .select("user_id_index", "pred_list")
    )

    return baseline_pred


def calculate_true_values(test_df):
    """
    return a true values in needed format
    :param test_df:
    :return:
    """
    items_for_user_true = (
        test_df.filter(test_df.rating > 4).groupBy("user_id_index")
        .agg(expr("collect_list(" + "anime_id_index" + ") as ground_truth"))
        .select("user_id_index", "ground_truth")
    )

    return items_for_user_true


def evaluate(name, pred, items_for_user_true):
    """
    Evaluate model prediction using MAP and NDCG
    :param name: name of model
    :param pred: predictions
    :param items_for_user_true: true values
    :return: None
    """
    items_for_user_all = pred.join(
        items_for_user_true, pred.user_id_index == items_for_user_true.user_id_index
    ).drop(pred.user_id_index)

    map_evaluations = RankingEvaluator(
        labelCol="ground_truth",
        predictionCol="pred_list",
        metricName="meanAveragePrecision",
        k=5
    )

    ndcg_evaluations = RankingEvaluator(
        labelCol="ground_truth",
        predictionCol="pred_list",
        metricName="ndcgAtK",
        k=5
    )

    map_at_5 = map_evaluations.evaluate(
        items_for_user_all
    )

    print name + " MAP at 5 = " + str(map_at_5)

    ndcg_at_5 = ndcg_evaluations.evaluate(
        items_for_user_all
    )

    print name + " NDCG at 5 = " + str(ndcg_at_5)


def make_pred_first_model(model, train_df, transformed):
    """
    make a prediction for ALS model and make it to prediction format
    :param model: trained ALS model
    :param train_df: train dataframe
    :param transformed: whole dataframe
    :return: prediction
    """
    users = transformed.select("user_id_index").distinct()
    items = transformed.select("anime_id_index").distinct()
    user_item = users.crossJoin(items)
    pred_df = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = pred_df.alias("pred").join(
        train_df.alias("train"),
        (pred_df["user_id_index"] == train_df["user_id_index"]) &
        (pred_df["anime_id_index"] == train_df["anime_id_index"]), how='outer')

    dfs_pred_final = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull())\
        .select('pred.' + "user_id_index", 'pred.' + "anime_id_index", 'pred.' + "prediction")

    items_for_user_pred = (
        dfs_pred_final.groupBy("user_id_index")
        .agg(F.sort_array(F.collect_list(F.struct("prediction", "anime_id_index")), asc=False)
             .alias("collected_list")
             )
        .withColumn("pred_list", F.col("collected_list.anime_id_index"))
        .drop("collected_list")
    )

    return items_for_user_pred


def first_model(train_df, max_iter=15, reg_param=0.09, rank=10):
    """
    train ALS model
    :param train_df: train dataframe
    :param max_iter: maximum iteration
    :param reg_param: regularization parameters
    :param rank: rank
    :return: trained model
    """
    als = ALS(maxIter=max_iter, regParam=reg_param, rank=rank,
              userCol="user_id_index",
              itemCol="anime_id_index",
              ratingCol="rating",
              coldStartStrategy="drop",
              nonnegative=True)
    model = als.fit(train_df)

    return model


def grid_search_first_model(train_df):
    """
    Grid search for ALS model
    :param train_df: train dataframe
    :return: best ALS model
    """
    als_model = ALS(maxIter=15, userCol="user_id_index",
                    itemCol="anime_id_index", ratingCol="rating",
                    coldStartStrategy="drop", nonnegative=True)
    params = ParamGridBuilder().addGrid(als_model.regParam, [0.001, 0.1, 1.0])\
        .addGrid(als_model.rank, [10, 15, 20]).build()
    evaluator = RegressionEvaluator(metricName='rmse',
                                    labelCol='rating',
                                    predictionCol='prediction')
    cross_validator = CrossValidator(estimator=als_model, estimatorParamMaps=params,
                                     evaluator=evaluator, numFolds=4, parallelism=4)
    best_model = cross_validator.fit(train_df)
    grid_model = best_model.bestModel

    print('Best Param (regParam): ', grid_model._java_obj.parent().getRegParam())
    print('Best Param (Rank): ', grid_model._java_obj.parent().getRank())

    return grid_model

