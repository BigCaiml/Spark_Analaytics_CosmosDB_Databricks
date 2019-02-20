# Databricks notebook source
cosmosdb_uri = 'INSERT COSMOSDB URI HERE'
cosmosdb_master_key = 'INSERT COSMOSDB READ-WRITE KEY HERE'

# COMMAND ----------

# MAGIC %md ####Step 1: Simulate Application Modification to Profiles
# MAGIC 
# MAGIC Our website application allows customers to modify their profiles.  It also creates new profiles for new customers. In the case of profile modifications, a customer's score may no longer be valid.  In the case of new profile creation, there may be no score at all.  We need to find these two types of profiles and get them scored using our trained model.
# MAGIC 
# MAGIC To do this, let's first simulate some modifications to existing profiles.  When the website modifies a profile, it updates the value assigned to the modified_dt key.  In a more sophisticated scenario, we might distinguish between changes that affect scores and those that don't.  But in this scenario, we'll keep it simple so that any modification will require a rescore.  To that end, all we need to do is update the modified_dt key on a few profiles to indicate they need rescoring: 

# COMMAND ----------

# retrieve ~50 semi-randomly selected profiles
read_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'query_custom' : 'select top 1000 * from profiles p',
  'query_pagesize' : '1000',
  'schema_samplesize': '1000'
  }
# 5% of 1000 profiles = ~50 profiles
profiles = spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**read_config).load().sample(fraction=0.05).cache()

from pyspark.sql.functions import current_timestamp
modified_profiles = profiles.withColumn('modified_dt', current_timestamp())

write_config =  {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'Upsert' : 'True',
  'WriteBatchSize' : 1000
  }
modified_profiles.repartition(8).write.format('com.microsoft.azure.cosmosdb.spark').options(**write_config).mode('overwrite').save()
print( '{0} profiles have been modified'.format(modified_profiles.count()) )

# COMMAND ----------

# MAGIC %md Now, let's create a few new profiles, leveraging existing profiles as templates. We'll need to make sure that we don't have any overlapping ids:

# COMMAND ----------

# get the max integer value associated with id's in the collection
read_maxid_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'query_custom' : 'SELECT TOP 1 right(p.id, length(p.id)-2) FROM profiles p ORDER BY p.id DESC'
  }
maxid_df = spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**read_maxid_config).load()
maxid = int( maxid_df.collect()[0]['$1'] )

# retrieve ~50 semi-randomly selected profiles
profiles = spark.read.format("com.microsoft.azure.cosmosdb.spark").options(**read_config).load().sample(fraction=0.05).cache()
profiles.createOrReplaceTempView('profiles')

# create new profiles from old ones, incrementing the int portion of the id and assigning a new modified_dt value
new_sql = '''
  select
    'AW' || lpad( cast(substr(id, 3) as int)+{0}, 8, '0') as id,
    firstpurchase,
    address,
    demographics,
    name,
    contact,
    current_timestamp as modified_dt
  from profiles
  '''.format(maxid)
new_profiles = spark.sql(new_sql)

# insert new profiles into collection
new_profiles.repartition(8).write.format('com.microsoft.azure.cosmosdb.spark').options(**write_config).mode('overwrite').save()
print( '{0} new profiles have been inserted'.format(new_profiles.count()) )

# COMMAND ----------

# MAGIC %md ####Step 2: Retrieve Profiles to Score/Re-Score
# MAGIC Read the maximum modified_dt recorded in our last scoring exericse:

# COMMAND ----------

checkpoint = spark.read.parquet('dbfs:/mnt/datasets/batch_score_checkpoint/').collect()[0]['dt']
print(checkpoint)

# COMMAND ----------

read_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'query_custom' : 'select * from profiles p where p.modified_dt > {0}'.format(checkpoint),
  'query_pagesize' : '1000',
  'schema_samplesize': '1000'
  }
changes = spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**read_config).load().cache()
print('{0} profiles have been retrieved which require rescoring'.format(changes.count()))

# COMMAND ----------

# MAGIC %md ####Step 3: Rescore Profiles
# MAGIC Here, we simply repeat the scoring work we did in the last lab.  For more info on the steps performed here, please review that lab:

# COMMAND ----------

# get required fields and apply type casting
demographics = changes.select(
  changes.id.alias('CustomerAlternateKey'), 
  changes.demographics.maritalstatus.alias('MaritalStatus'),
  changes.demographics.gender.alias('Gender'),
  changes.demographics.yearlyincome.cast('double').alias('YearlyIncome'),
  changes.demographics.totalchildren.cast('integer').alias('TotalChildren'),
  changes.demographics.numberchildrenathome.cast('integer').alias('NumberChildrenAtHome'),
  changes.demographics.education.alias('Education'),
  changes.demographics.occupation.alias('Occupation'),
  changes.demographics.houseownerflag.cast('integer').alias('HouseOwnerFlag'),
  changes.demographics.numbercarsowned.cast('integer').alias('NumberCarsOwned'),
  changes.demographics.commutedistance.alias('CommuteDistance'),
  changes.demographics.region.alias('Region'),
  changes.demographics.age.cast('integer').alias('Age')
  )

# import required libraries
from mmlspark import ComputeModelStatistics, TrainedClassifierModel
scoring_model = TrainedClassifierModel.load('dbfs:/mnt/datasets/model/bikeBuyer.mml')
scored_output = scoring_model.transform( demographics )

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType

def vector_to_array( col ):
  return udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))(col)

# retrieve id and score (array) from output dataset
scores = scored_output.select( 
    scored_output['CustomerAlternateKey'].alias('id'), 
    vector_to_array( scored_output.scored_probabilities ).alias('score')
    )

# assemble scored profile
changes.registerTempTable('profiles')
scores.registerTempTable('scores')

sql = '''
  SELECT
    a.*,
    b.scores
  FROM profiles a
  JOIN (
    select
      id,
      named_struct('bike_buyer', score[1]) as scores    
    from scores
    ) b
    ON a.id=b.id
  '''
to_publish = spark.sql(sql).repartition(8)
to_publish.write.format('com.microsoft.azure.cosmosdb.spark').options(**write_config).mode('overwrite').save()
print( '{0} profiles have been scored and sent back to CosmosDB'.format(to_publish.count()) )

# COMMAND ----------

# MAGIC %md ####Step 4: Increment the Checkpoint Data for Next Run
# MAGIC Now, let's get the max modified_dt from this last run.  This will be checkpoint for our next scoring cycle.

# COMMAND ----------

spark.sql('select max(modified_dt) as dt from profiles').write.parquet('dbfs:/mnt/datasets/batch_score_checkpoint/', mode='overwrite')

# COMMAND ----------

# MAGIC %md ####Sidebar: What about using the CosmosDB Change Log?
# MAGIC 
# MAGIC The CosmosDB change feed keeps track of which documents in a CosmosDB collection have been newly created or modified. When you query the change feed, you retrieve the current state of the document.  If you are familiar with Change Data Capture and Change Tracking in SQL Server, the CosmosDB change feed conceptually mirrors the Change Tracking feature.
# MAGIC 
# MAGIC Here is one example of the change feed being read from our Spark environment:

# COMMAND ----------

# read from the change feed, setting a checkpoint
cf_read_feed_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles', 
  'ReadChangeFeed' : 'True',                                  # indicates that we are reading the change feed 
  'ChangeFeedQueryName' : 'incremental_profile_scoring',      # provides a name for the reader which will also be used to name the checkpoint file
  'ChangeFeedCheckpointLocation' : 'dbfs:/mnt/datasets/changefeed/',  # location in which to store the checkpoint file
  'ChangeFeedStartFromTheBeginning' : 'False',                # indicates that the reader (named by the ChangeFeedQueryName property) should not restart at beginning
  'ChangeFeedUseNextToken': 'True',                           # indicates that we should read next increment given available checkpoint info
  'RollingChangeFeed': 'True'                                 # indicates that the change feed can cycle over itself (if needed) between reads
  }

cf_changes = spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**cf_read_feed_config).load()
cf_changes.count()

# COMMAND ----------

# MAGIC %md If you ran the last block of code, you may have noticed that no documents came across.  That's because when you initialize a change feed read with ChangeFeedStartFromTheBeginning set to False, you are capturing a checkpoint for the current state of the feed. You can see that checkpoing information in the folder identified by the ChangeFeedCheckpointLocation property.  If you make a change and then re-query the feed, the documents associated with those changes will now come across.  Here, we modify one profile by updating its modified_dt value:

# COMMAND ----------

cf_read_config = {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'query_custom' : 'select top 1 * from profiles p',
  'query_pagesize' : '1000',
  'schema_samplesize': '1000'
  }
cf_profiles = spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**cf_read_config).load().cache()

from pyspark.sql.functions import current_timestamp
cf_modified_profiles = cf_profiles.withColumn('modified_dt', current_timestamp())

cf_write_config =  {
  'Endpoint' : cosmosdb_uri,
  'Masterkey' : cosmosdb_master_key,
  'Database' : 'app',
  'Collection' : 'profiles',
  'Upsert' : 'True',
  'WriteBatchSize' : 1000
  }
cf_modified_profiles.write.format('com.microsoft.azure.cosmosdb.spark').options(**cf_write_config).mode('overwrite').save()
print( '{0} profiles have been modified'.format(cf_modified_profiles.count()) )

# COMMAND ----------

# MAGIC %md Query the change feed now and you see that one document on the feed:

# COMMAND ----------

display(cf_changes)

# COMMAND ----------

# MAGIC %md Query it again and notice that checkpoint has now moved past that change:

# COMMAND ----------

display(cf_changes)

# COMMAND ----------

# MAGIC %md If the mechanics of the change feed now make a little more sense, think about how you would use this to accomplish incremental data processing like we've demonstrated in this lab. When the app updates a profile or creates a new profile, that profile will show up on the next pull from the feed.  If you retrieve those profiles, score them and then add them back to the collection, they show up again on the next pull.  This could be managed if you knew the application that triggered the initial changes was quiesced during the scoring run (which would include a step to move the checkpoing past the updates associated with the scoring). But this doesn't seem like the right solution.  Instead, a soft-delta mechanism, such as the one demonstrated in this lab, seems to be a more reliable approach (assuming you can get the app to reliably update the modified_dt key).
# MAGIC 
# MAGIC So when is the change feed best to use?  Much like SQL Server's Change tracking feature, the change feed is great when you are incrementally pulling changes for downstream data modifictions, whether batch or streaming.
# MAGIC 
# MAGIC With regard to streaming, notice in the configuration settings above a property named ChangeFeedStartFromTheBeginning. If you are performing batch pulls from the change feed, this setting will cause your pull to read the change feed in its entirity between sessions?  What's a session?  Each batch job would behave as its own session.  Saving checkpoint data to disk has no impact on the behavior.  When the batch job terminates and then restarts, the next pull starts from the begginning again. Only set this property to True when you desire this behavior which is typical when you will have a routine that runs continously.
