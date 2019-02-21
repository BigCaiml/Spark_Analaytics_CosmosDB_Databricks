# Databricks notebook source
# MAGIC %md In this notebook, you will simply mount the datasets container in the Azure Storage Account you just created to the Databricks file system. This will require you to retrieve a storage key from the Azure Portal which you will paste in clear text below.  In a production environment, you would not want to expose your storage key in this manner and instead should make use of [Databricks Secrets](https://docs.azuredatabricks.net/user-guide/secrets/index.html "Documentation on Databricks Secrets") which will hide the key value from users of the Databricks environment.

# COMMAND ----------

# copy and paste the required values from the azure portal
account_name = 'INSERT STORAGE ACCOUNT NAME HERE'  
account_key = 'INSERT STORAGE ACCOUNT KEY HERE'

# mount your azure storage account to the databricks file system
try:
  dbutils.fs.mount(
    source = 'wasbs://datasets@{0}.blob.core.windows.net'.format(account_name),
    mount_point = '/mnt/datasets',
    extra_configs = {'fs.azure.account.key.{0}.blob.core.windows.net'.format(account_name):account_key}
    )
except:
  pass

# COMMAND ----------

# MAGIC %md Download data files from the github repository into your mounted storage account.

# COMMAND ----------

# MAGIC %sh wget -O /dbfs/mnt/datasets/training/purchases.txt 'https://github.com/BigCaiml/Spark_Analytics_CosmosDB_Databricks/raw/master/datasets/history.txt'

# COMMAND ----------

# MAGIC %sh wget -O /dbfs/mnt/datasets/profiles/profiles.txt 'https://github.com/BigCaiml/Spark_Analytics_CosmosDB_Databricks/raw/master/datasets/profiles.txt' 

# COMMAND ----------

# MAGIC %md Use the following two cells to verify the files have landed successfully in your file system

# COMMAND ----------

# MAGIC %fs head dbfs:/mnt/datasets/training/purchases.txt

# COMMAND ----------

# MAGIC %fs head dbfs:/mnt/datasets/profiles/profiles.txt
