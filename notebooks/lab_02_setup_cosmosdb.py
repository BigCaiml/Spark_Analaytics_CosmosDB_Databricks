# Databricks notebook source
# MAGIC %md #####Before running this notebook...
# MAGIC You must install the CosmosDB Connector for Spark per the steps in [this document](https://docs.azuredatabricks.net/spark/latest/data-sources/azure/cosmosdb-connector.html).  Be sure you download the version of the Connector which aligns with the version of Spark in use by your cluster.  When you are done with the installation, be sure to restart your cluster. 

# COMMAND ----------

# MAGIC %md Copy and paste values for your CosmosDB account (under Keys in the portal) into the cell below. As before, we would not typically paste sensitive values in a notebook like this and instead would make use of the [Databricks Secrets feature](https://docs.azuredatabricks.net/user-guide/secrets/index.html) to hide these from users.

# COMMAND ----------

cosmosdb_uri = 'INSERT COSMOSDB URI HERE'
cosmosdb_master_key = 'INSERT COSMOSDB READ-WRITE KEY HERE'  

# COMMAND ----------

# MAGIC %md In the next cell, we will read the data we will use to construct our profile JSON documents from a tab-delimited text file:

# COMMAND ----------

# read data from profiles.txt into temp view
spark.read.csv('dbfs:/mnt/datasets/profiles/profiles.txt', header=True, sep='\t').createOrReplaceTempView('profiles')

# COMMAND ----------

# MAGIC %md Now we will restructure this data to reflect some of the more complex nesting structures commonly found in JSON. Notice that each document has an id field which will be used to uniquely identify the document.  Also, each field has a modified_dt field which we are imagining is set by the website application when a profile is created or modified (by the site):

# COMMAND ----------

# reorganize tabular data from profiles.txt into required structure
sql_statement ='''
  SELECT
    CustomerAlternateKey as id,
    firstpurchase,
    named_struct(
      'first', first,
      'middle', middle,
      'last', last) as name,
    named_struct(
      'email', email,
      'phone', phone
      ) as contact,
    named_struct(
      'line1', line1,
      'line2', line2,
      'city', city,
      'stateprovince', stateprovince,
      'country', country
      ) as address,
    named_struct(
      'maritalstatus', maritalstatus,
      'gender', gender,
      'yearlyincome', yearlyincome,
      'totalchildren', totalchildren,
      'numberchildrenathome', numberchildrenathome,
      'education', education,
      'occupation', occupation,
      'houseownerflag', houseownerflag,
      'numbercarsowned', numbercarsowned,
      'commutedistance', commutedistance,
      'region', region,
      'age', age
      ) as demographics,
      current_timestamp as modified_dt
  FROM profiles'''

# review query results
display(spark.sql(sql_statement))

# COMMAND ----------

# MAGIC %md Now we are writing our data to CosmosDB to initialize the lab environment:

# COMMAND ----------

# connection info for cosmosdb connector
config =  {
'Endpoint' : cosmosdb_uri,
'Masterkey' : cosmosdb_master_key,
'Database' : 'app',
'Collection' : 'profiles',
'Upsert': True
  }

# deploy profiles to cosmosdb
df = spark.sql(sql_statement).repartition(2) 
df.write.format('com.microsoft.azure.cosmosdb.spark').options(**config).mode('overwrite').save()

# COMMAND ----------

# MAGIC %md Please verify that the query below shows that your CosmosDB collection now contains 18,484 profile documents. **If you receive a 429 error**, *i.e.* "Request rate is large", or if the count doesn't quite equal the expected value, please wait a few seconds and re-run the next cell again.

# COMMAND ----------

config['query_custom'] = 'SELECT COUNT(1) as profile_count FROM profiles p'
display(spark.read.format('com.microsoft.azure.cosmosdb.spark').options(**config).load())

# COMMAND ----------


