import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable  # avoids "Python worker failed" on Windows

from pyspark.sql import SparkSession, functions as F


def build_features(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.drop("RowNumber", "CustomerId", "Surname")  # identifiers, no signal

    df = (
        df
        .withColumn("BalanceSalaryRatio", F.col("Balance") / (F.col("EstimatedSalary") + 1))
        .withColumn("ZeroBalance", (F.col("Balance") == 0).cast("int"))
        .withColumn("TenureByAge", F.col("Tenure") / F.col("Age"))
        .withColumn(
            "AgeGroup",
            F.when(F.col("Age") < 30, "<30")
             .when(F.col("Age") < 40, "30-39")
             .when(F.col("Age") < 50, "40-49")
             .when(F.col("Age") < 60, "50-59")
             .otherwise("60+"),
        )
        .withColumn(
            "ProductGroup",
            F.when(F.col("NumOfProducts") == 1, "1")
             .when(F.col("NumOfProducts") == 2, "2")
             .otherwise("3+"),
        )
        .withColumn(
            "InactiveWithBalance",
            ((F.col("IsActiveMember") == 0) & (F.col("Balance") > 0)).cast("int"),
        )
    )
    return df


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("churn-features").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    feats = build_features(spark, "data/Churn_Modelling.csv")
    feats.printSchema()

    pdf = feats.toPandas()
    pdf.to_parquet("data/features.parquet", index=False)
    print("Saved:", pdf.shape)
    print(pdf.head())

    spark.stop()