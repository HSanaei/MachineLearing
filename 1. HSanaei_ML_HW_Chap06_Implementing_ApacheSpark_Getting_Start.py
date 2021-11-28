
"""The entry point to a Spark program is creating a Spark session, which can be done
by using the following lines:"""

from pyspark.sql import SparkSession
spark = SparkSession \
            .builder \
            .appName("test") \
            .getOrCreate()


""" In the following example, we will create
a DataFrame object, df, from a CSV file:"""
 
df = spark.read.csv("file:///Spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.csv", header=True, sep=';')

""" display the contents of the df object by using the following command:"""
df.show()

df.count()

df.printSchema()

df.select("name").show()
df.select(["name", "job"]).show()

""" We can filter rows by condition, for instance, by the value of one column, using the
following command:"""

df.filter(df['age'] > 31).show()



from pyspark.sql.functions import monotonically_increasing_id
df.withColumn('index', monotonically_increasing_id())