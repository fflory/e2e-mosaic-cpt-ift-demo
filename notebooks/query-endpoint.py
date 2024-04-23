# Databricks notebook source
QA_TEMPLATE_ZEROSHOT = """
You are a Regulatory Reporting Assistant. 
Please answer the question as precise as possible.  
If you do not know, just say I don't know.

### Instruction:
Please answer the question:
-- Question
{question}
------

### Response:
"""

QA_TEMPLATE_WITH_CTX = """You are a Regulatory Reporting Assistant. 
Please answer the question as precise as possible using information in context. 
If you do not know, just say I don't know.

### Instruction:
Please answer the question using the given context:
-- Context:
{context}
------
-- Question
{question}
------

### Response:
"""

# COMMAND ----------

model_name = ["mistral7bv01_ift_v1", "ift-mistral-7b-v0-1-vpdi1t", "llam7b-ift-v1", "doan_mistral_7b_ift"][0]

# COMMAND ----------

prompt = QA_TEMPLATE_WITH_CTX.format(**spark.table("fflory.finreg.qa_dataset_val").take(1)[0].asDict())
print(prompt)

# COMMAND ----------

import os
import mlflow
from mlflow.deployments import get_deploy_client
import pandas as pd
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import ArrayType, StringType

client = mlflow.deployments.get_deploy_client("databricks")

chat_response = client.predict(
    endpoint=model_name,
    inputs={
        "prompt": prompt,
        "temperature": 0.1,
        "max_tokens": 500,
        "n": 1,
    },
)

# COMMAND ----------

print(chat_response["choices"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference with spark UDFs

# COMMAND ----------

display(spark.table("fflory.finreg.qa_dataset_val").limit(4))

# COMMAND ----------

import pandas as pd
from mlflow.deployments import get_deploy_client
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import mlflow

def predict_with_foundation_model(model_name):
    """
    Performs prediction using the foundation model.

    Args:
    - model_name (str): The name of the model endpoint.
    - prompt (str): removed The prompt for the prediction.

    Returns:
    predict_with_model(pandas_udf): pandas udf that can be called incoming dataset
    """
    @pandas_udf("string")
    def predict_with_model(text_series: pd.Series) -> pd.Series:
        predictions = []

        client = mlflow.deployments.get_deploy_client("databricks")

        for text in text_series:
            chat_response = client.predict(
                endpoint=model_name,
                # inputs={
                #     "messages": [
                #         {
                #             "role": "user",
                #             "content": f""" INSTRUCTION: {prompt}
                #             ARTICLE: {text}
                #             """,
                #         },
                #     ],
                #     "temperature": 0.1,
                #     "max_tokens": 750,
                # },
                inputs={
                    "prompt": text, # prompt + article,
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "n": 1,
                },
            )

            # predictions.append(chat_response["choices"][0]["message"]["content"])
            predictions.append(chat_response["choices"][0]["text"])

        return pd.Series(predictions)
    return predict_with_model

# Usage in a Databricks notebook or job
predict_udf = predict_with_foundation_model(model_name)

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

display(spark.table("fflory.finreg.qa_dataset_val").limit(4).withColumn("generated_answer", predict_udf(F.col("question"))))

# COMMAND ----------

checkpoint_location = "/tmp/fflory/summarization_pipeline/dbrx500/1"

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists fflory.finreg.qa_dataset_output

# COMMAND ----------

# !rm -rf /tmp/fflory/summarization_pipeline/dbrx500/1

# COMMAND ----------

from pyspark.sql.functions import col, lit
input_stream_repartitioned = spark.readStream.option("maxFilesPerTrigger", 1).table("fflory.finreg.qa_dataset_val")

input_stream_summarized = input_stream_repartitioned\
  .withColumn("generated_answer", predict_udf(col("question")))\
  .withColumn("model", lit(model_name))  #include a column to track the model/endpoint that generated summarization for lineage

input_stream_summarized.writeStream.outputMode("append")\
  .trigger(processingTime="1 second")\
  .option("maxFilesPerTrigger", 1)\
  .option("skipChangeCommits", "true")\
  .option("checkpointLocation", checkpoint_location)\
  .toTable(f"fflory.finreg.qa_dataset_output")

# COMMAND ----------

display(spark.table("fflory.finreg.qa_dataset_output").limit(10))
