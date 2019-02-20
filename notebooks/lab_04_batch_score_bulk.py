# Databricks notebook source
cosmosdb_uri = 'INSERT COSMOSDB URI HERE'
cosmosdb_master_key = 'INSERT COSMOSDB READ-WRITE KEY HERE'

# COMMAND ----------

# MAGIC %md ####Step 1: Read Profiles from CosmosDB
# MAGIC In the next cell, you will retrieve all profiles from your CosmosDB database. This might be the action we take when we first score our profiles or when we deploy a new predictive model.</p>
# MAGIC 
# MAGIC Notice that the <code>query_custom</code> configuration setting on the CosmosDB connector allows us to define which profiles to retrieve using a CosmosDB SQL statement. If we had an excessively large number of profiles to score, we might define a WHERE clause on this query to limit which profiles are returned.  In our lab scenario, we are dealing with less than 20,000 profiles, each of which is itself a small JSON document, so that we will simply grab them all.</p>
# MAGIC 
# MAGIC Notice too that the <code>query_pagesize</code> configuration setting controls the number of documents sent from CosmosDB to a Spark partition in one roundtrip. Per [the CosmosDB Spark Connector's source code](https://github.com/Azure/azure-cosmosdb-spark/blob/2.3/src/main/scala/com/microsoft/azure/cosmosdb/spark/config/CosmosDBConfig.scala), results are returned, by default, 50 documents at a time. You may consider elevating this value to improve throughput from CosmosDB to Spark.
# MAGIC 
# MAGIC Finally, notice the <code>schema_samplesize</code> setting determines how many documents will be read by Spark in order to infer a schema for the data frame. You can explicitly provide this schema using a pyspark.sql.type StructType assigned using [the schema() property](http://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.schema) of the DataframeReader, thereby disabling schema inference. Otherwise, set schema_samplesize to a value sufficient to ensure Spark sees a sufficient number of document variants to accurately infer a schema. The default setting for this configuration option is the default query pagesize, *i.e.* [50 documents](https://github.com/Azure/azure-cosmosdb-spark/blob/2.3/src/main/scala/com/microsoft/azure/cosmosdb/spark/config/CosmosDBConfig.scala).

# COMMAND ----------

# retrieve profiles from cosmosdb
read_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'query_pagesize' : '10000',
  'query_custom' : 'SELECT * FROM profiles p',
  'schema_samplesize': '10'
}

profiles = spark.read.format("com.microsoft.azure.cosmosdb.spark").options(**read_config).load().cache()
display(profiles)

# COMMAND ----------

# MAGIC %md ####Step 2: Generate Scores
# MAGIC In this step, we will pass a Spark dataframe to our ML model to generate a score for each record.  Our first challenge will be to determine what the structure of that dataframe should be. Unfortunately, there is no programmatic as of now to do this.  **You will need to speak with the Data Scientist to understand the field names and data types to be used in the dataframe.** 
# MAGIC 
# MAGIC With that in mind, we determine through our Data Scientist that the model expects an input dataframe with the following fields:</p>
# MAGIC * CustomerAlternateKey: string
# MAGIC * MaritalStatus: string
# MAGIC * Gender: string
# MAGIC * YearlyIncome: double
# MAGIC * TotalChildren: integer
# MAGIC * NumberChildrenAtHome: integer
# MAGIC * Education: string
# MAGIC * Occupation: string
# MAGIC * HouseOwnerFlag: integer
# MAGIC * NumberCarsOwned: integer
# MAGIC * CommuteDistance: string
# MAGIC * Region: string
# MAGIC * Age: integer
# MAGIC 
# MAGIC Note that the CustomerAlternateKey field aligns with the id field in our profiles documents. This field is passed to the model but not used to actually generate scores.  By including this field in the input, we ensure that it is returned in the scored output.  We can then use this field to match scores with profiles as part of our document update.
# MAGIC 
# MAGIC Now that we understand the data structure our model expects, take a quick look at the demographics data in the output displayed in the cell above.  Notice that the demographics data is nested within the document. We'll need to "flatten" that structure to build our required model input. Also, all the demographic information originating from our JSON documents are being typed as strings. We'll need to explicitly cast the non-string fields to the required types.

# COMMAND ----------

# get required fields and apply type casting
demographics = profiles.select(
  profiles.id.alias('CustomerAlternateKey'), 
  profiles.demographics.maritalstatus.alias('MaritalStatus'),
  profiles.demographics.gender.alias('Gender'),
  profiles.demographics.yearlyincome.cast('double').alias('YearlyIncome'),
  profiles.demographics.totalchildren.cast('integer').alias('TotalChildren'),
  profiles.demographics.numberchildrenathome.cast('integer').alias('NumberChildrenAtHome'),
  profiles.demographics.education.alias('Education'),
  profiles.demographics.occupation.alias('Occupation'),
  profiles.demographics.houseownerflag.cast('integer').alias('HouseOwnerFlag'),
  profiles.demographics.numbercarsowned.cast('integer').alias('NumberCarsOwned'),
  profiles.demographics.commutedistance.alias('CommuteDistance'),
  profiles.demographics.region.alias('Region'),
  profiles.demographics.age.cast('integer').alias('Age')
)
display(demographics)

# COMMAND ----------

# MAGIC %md With the required input dataset assembled, let's score each entry using our model. Notice as mentioned above that the output dataframe includes all input fields plus score-related fields generated by our model.

# COMMAND ----------

# import required libraries
from mmlspark import ComputeModelStatistics, TrainedClassifierModel

# load the trained model
scoring_model = TrainedClassifierModel.load('dbfs:/mnt/datasets/model/bikeBuyer.mml')

# score the input data
scored_output = scoring_model.transform( demographics )
display(scored_output)

# COMMAND ----------

# MAGIC %md We now have a scored dataset.  The scores themselves are presented in three fields:</p>
# MAGIC * **scores** - the raw parameters of a logistic model</p>
# MAGIC * **scored_probabilities** - the calculated probability of being in the first class (0 aka Non Bike Buyer) or second class (1 or Bike Buyer)</p>
# MAGIC * **scored_labels** - the predicted class based on scored_probabilities (0 or 1)</p>
# MAGIC 
# MAGIC Of the data returned by the model, the only value we are concerned with is the probability of being a Bike Buyer.  This is the second probability presented in the scored_probabilities field. Notice that the combined probabilities of being a Non Bike Buyer and a Bike Buyer add up to 1.0. Knowing one probability allows you to easily calculate the other, hence our need to grab just the one probability for being a Bike Buyer.

# COMMAND ----------

scored_output.printSchema()

# COMMAND ----------

# MAGIC %md You probably noticed that the scored_probabilities field (and the scores field) are presented with an unusual looking structure. Looking at the schema of the output dataframe, printed in the cell above, you can see that Spark recognizes the field as having a user-defined type (udt).
# MAGIC 
# MAGIC That user-defined type is what is known in the Spark ML world as a vector type. A vector can be covnerted into a numpy.ndarray type which can then be converted to a standard list. The standard list can then be converted by SparkSQL into a SparkSQL array.  While that sounds like a lot of hoops to jump through, we've got to convert the vector into something that the CosmosDB Spark Connector can translate into JSON. 
# MAGIC 
# MAGIC **NOTE** In the code below, we are trying to explicitly handle the conversion of the score vector into an array using standard practices.  We could also brute-force it by simply casting the vector to a string and spliting the resulting string on it's comma-delimiter. Those more comfortable with SQL might find this easier to read even if it is a bit less standard. A commented out version of this logic is provided in the cell below for comparison purposes.

# COMMAND ----------

#from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType

# define function to convert vector field values into an array of double-precision floats
def vector_to_array( col ):
  return udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))(col)

# retrieve id and score (array) from output dataset
scores = scored_output.select( 
    scored_output['CustomerAlternateKey'].alias('id'), 
    vector_to_array( scored_output.scored_probabilities ).alias('score')
    )

## brute-force method
## -----------------------------------------------------------
#from pyspark.sql.functions import expr
#scores = scored_output.withColumn('score_string', scored_output.scored_probabilities.cast('string')).\
#  select(
#        scored_output['CustomerAlternateKey'].alias('id'),
#        expr("split(substring(score_string, 2, length(score_string)-2), ',')").cast('array<double>').alias('score')
#        )
## -----------------------------------------------------------

# review dataset
display(scores)
  

# COMMAND ----------

# MAGIC %md ####Step 3: Update Profiles in CosmosDB
# MAGIC Now that we have our scores for each profile, it's time to add these to the profile documents and submit these to CosmosDB for update.
# MAGIC 
# MAGIC To manipulate the profiles dataframe, we could continue to make use of the SQL API.  But personally, I find working with SQL statements is easier for this kind of work.  To enable this work, we'll start by registering our profiles and scores dataframes as temp tables:

# COMMAND ----------

# assemble scored profile
profiles.registerTempTable('profiles')
scores.registerTempTable('scores')

# COMMAND ----------

# MAGIC %md Now, we can issue a SQL statement that joins these two data sets to create the updated profile structure. Notice that the score field in the scores temp table is an array.  We are interested in the second score in the array (which uses a leading index of 0). This score that tells us the probability the customer associated with this profile is a bike buyer. 

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view scored_profiles as
# MAGIC   SELECT
# MAGIC     a.*,
# MAGIC     b.scores
# MAGIC   FROM profiles a
# MAGIC   JOIN (
# MAGIC     select
# MAGIC       id,
# MAGIC       named_struct('bike_buyer', score[1]) as scores    
# MAGIC     from scores
# MAGIC     ) b
# MAGIC     ON a.id=b.id;
# MAGIC     
# MAGIC select * from scored_profiles;

# COMMAND ----------

# MAGIC %md Why did we assign the bike buyer score as part of a nest of data under a scores property? It's foreseeable that if scoring for propensity to buy a bike has a positive business impact, we could calculate other scores for our customers, *e.g.* propensity to buy branded clothing, propensity to respond to free shipping, etc. Adding the scores to the profile in this manner provides us a structure for easily absorbing additional scores down the road. 

# COMMAND ----------

# MAGIC %md Now we can take the data from this temporary view and push it to CosmosDB.  We will do this using an Upsert mechanism by which incoming documents replace existing documents when their /id fields match.
# MAGIC 
# MAGIC </p>Notice that before writing the data to CosmosDB, we explicitly set the number of partitions to assign to the dataframe. The lesser of either the number of cores across our cluster's worker nodes or the number of partitions in our source dataframe determines the number of parallel writes Spark can perform to CosmosDB via the CosmosDB Connector.  There's no exact science in deciding how many parallel writes you should perform, but you need to be aware that too many parallel writes will exceed the number of RU/s that you've allocated to CosmosDB and will trigger a 429 error.  </p>If you experience a 429 error, try reducing the number of partitions you use with the dataframe and/or elevating the number of RU/s allocated to your CosmosDB collection, even if just temporarily.

# COMMAND ----------

write_config =  {
'Endpoint' : cosmosdb_uri,
'Masterkey' : cosmosdb_master_key,
'Database' : 'app',
'Collection' : 'profiles',
'Upsert' : 'true',
'WriteBatchSize' : 2000
  }

# assemble dataframe to publish, repartition to limit rate of flow to CosmosDB
to_publish = spark.sql('select * from scored_profiles').repartition(8)

# write the data to CosmosDB
to_publish.write.format('com.microsoft.azure.cosmosdb.spark').options(**write_config).mode('overwrite').save()

# COMMAND ----------

# MAGIC %md ####Step 4: Capture Last Modified_DT Value for Incremental Updates
# MAGIC With all of our profiles scored, we now should capture the latest modified_dt value found in those profiles.  This value is set by our app when a profile is created or modified. By capturing this, we can create a checkpoint for future incremental updates.

# COMMAND ----------

spark.sql('select max(modified_dt) as dt from profiles').write.parquet('dbfs:/mnt/datasets/batch_score_checkpoint/', mode='overwrite')
