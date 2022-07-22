from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.sparkContext.parallelize([(1, 2, 3, 'a b c'),
    (4, 5, 6, 'd e f'),
    (7, 8, 9, 'g h i')]).toDF(['col1', 'col2', 'col3','col4'])

df.show()

|col1|col2|col3| col4|
+----+----+----+-----+
| 1| 2| 3|a b c|
| 4| 5| 6|d e f|
| 7| 8| 9|g h i|
+----+----+----+-----+


from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

myData = spark.sparkContext.parallelize([(1,2), (3,4), (5,6), (7,8), (9,10)])

myData.collect()

[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# createDataframe( ) function

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

Employee = spark.createDataFrame([
                    ('1', 'Joe', '70000', '1'),
                    ('2', 'Henry', '80000', '2'),
                    ('3', 'Sam', '60000', '2'),
                    ('4', 'Max', '90000', '1')],
                    ['Id', 'Name', 'Sallary','DepartmentId']
                    )

+---+-----+-------+------------+
| Id| Name|Sallary|DepartmentId|
+---+-----+-------+------------+
| 1| Joe| 70000| 1|
| 2|Henry| 80000| 2|
| 3| Sam| 60000| 2|
| 4| Max| 90000| 1|
+---+-----+-------+------------+



