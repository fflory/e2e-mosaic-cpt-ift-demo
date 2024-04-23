# Databricks notebook source
# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

# os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("dbdemos", "azure-openai")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

from openai import OpenAI


DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
        
client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)
        
completions = client.completions.create(
  prompt='Write 3 reasons why you should train an AI model on domain specific data sets?',
  model="ift-mistral-7b-v0-1-vpdi1t",
  max_tokens=528
)
         
print(completions.choices[0].text)

# COMMAND ----------

# %sql
# SELECT ai_query(
#   'ift-mistral-7b-v0-1-vpdi1t',
#   named_struct("prompt","What is Machine Learning?")
#   -- returnType => schema_of_json('@outputJson')
# )

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

prompt = "what is machine learning"

# COMMAND ----------

import os
import mlflow
from mlflow.deployments import get_deploy_client
import pandas as pd
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import ArrayType, StringType

client = mlflow.deployments.get_deploy_client("databricks")

chat_response = client.predict(
    endpoint="ift-mistral-7b-v0-1-vpdi1t ",
    inputs={
        "prompt": prompt,# + article,
        "temperature": 0.1,
        "max_tokens": 500,
        "n": 1,
    },
)

# COMMAND ----------

print(chat_response)

# COMMAND ----------

# MAGIC %pip install databricks_genai_inference
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from databricks_genai_inference import ChatSession
# https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/databricks-mixtral-8x7b-instruct/invocations

# the custom serving endpoint is not compatible with the ChatSession inference format
_ep = "ift-mistral-7b-v0-1-vpdi1t"
# _ep = "databricks-mixtral-8x7b-instruct"
chat = ChatSession(model=_ep, system_message="You are a helpful assistant.", max_tokens=128)
chat.reply("Knock, knock!")
chat.last # return "Hello! Who's there?"
chat.reply("Guess who!")
chat.last # return "Okay, I'll play along! Is it a person, a place, or a thing?"

chat.history
# return: [
#     {'role': 'system', 'content': 'You are a helpful assistant.'},
#     {'role': 'user', 'content': 'Knock, knock.'},
#     {'role': 'assistant', 'content': "Hello! Who's there?"},
#     {'role': 'user', 'content': 'Guess who!'},
#     {'role': 'assistant', 'content': "Okay, I'll play along! Is it a person, a place, or a thing?"}
# ]

