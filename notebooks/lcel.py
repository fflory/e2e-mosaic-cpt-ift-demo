# Databricks notebook source
# %pip install openai
%pip install --upgrade --quiet  langchain-core langchain-community langchain-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("dbdemos", "azure-openai")
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("felix-flory", "oai")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="ift-mistral-7b-v0-1-vpdi1t")

# COMMAND ----------

model.invoke("what is spark")

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})

# COMMAND ----------



# COMMAND ----------

# start over
from langchain_community.chat_models.databricks import ChatDatabricks
llm_mistral = ChatDatabricks(targe_uri="databricks", endpoint="ift-mistral-7b-v0-1-vpdi1t", temperature=0.1)
llm_mistral.invoke("what is apache spark")

# COMMAND ----------

model = llm_mistral

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})

# COMMAND ----------

llm_mistral.invoke("what is spark")

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

chat = ChatDatabricks(
    # target_uri="databricks",
    endpoint="ift-mistral-7b-v0-1-vpdi1t",
    temperature-0.1,
)

# single input invocation
print(chat_model.invoke("What is MLflow?").content)

# single input invocation with streaming response
for chunk in chat_model.stream("What is MLflow?"):
    print(chunk.content, end="|")
