from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

rf = RandomForestClassifier(n_jobs=-1)
k_fold = KFold(n_splits=5)
cross_val_score(rf, X_features, data['label'], cv=k_fold, scoring='accuracy', n_jobs=-1)

########################################################################################

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.sparkContext\
    .parallelize([(1, 2, 3, 'a b c'),
        (4, 5, 6, 'd e f'),
        (7, 8, 9, 'g h i')])\
    .toDF(['col1', 'col2', 'col3','col4'])

df.show()

