# Databricks notebook source
# MAGIC %md
# MAGIC at the end of this notebook is a working example

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------



# COMMAND ----------

DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

# COMMAND ----------

from openai import OpenAI
import os
        
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
        
client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)
        
completions = client.completions.create(
  prompt='Write 3 reasons why you should train an AI model on domain specific data sets?',
  model="ff_finreg_llama7b_ift",
  max_tokens=128
)
         
print(completions.choices[0].text)

# COMMAND ----------

# MAGIC %pip install databricks_genai_inference

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


from databricks_genai_inference import ChatSession

chat = ChatSession(model="ff_finreg_llama7b_ift", system_message="You are a helpful assistant.", max_tokens=128)
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


# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


import os
import openai
from openai import OpenAI

client = OpenAI(
    api_key="dapi-your-databricks-token",
    base_url="https://example.staging.cloud.databricks.com/serving-endpoints"
)

response = client.embeddings.create(
  model="databricks-bge-large-en",
  input="what is databricks"
)


# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("dbdemos", "azure-openai")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

from openai import OpenAI
import os
        
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
        
client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)
        
completions = client.completions.create(
  prompt='Write 3 reasons why you should train an AI model on domain specific data sets?',
  model=["llam7b-ift-v1", "doan_mistral_7b_ift"][0],
  max_tokens=128
)
         
print(completions.choices[0].text)
