# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain_community.chat_models.databricks import ChatDatabricks

from finreganalytics.dataprep.qagen import build_qa_eval_dataset
from finreganalytics.utils import get_spark
from finreganalytics.dataprep import store_as_mds, store_as_jsonl
from finreganalytics.dataprep.ift_data_prep import prepare_ift_dataset

try:
    context = dbutils.entry_point.getDbutils().notebook().getContext()  # noqa
    os.environ["DATABRICKS_HOST"] = context.apiToken().get()
    os.environ["DATABRICKS_TOKEN"] = context.apiUrl().get()
except:
    pass

# COMMAND ----------

spark.read.table("msh.finreg.qa_dataset").write.saveAsTable("fflory.finreg.qa_dataset_2")

# COMMAND ----------



# COMMAND ----------

mds_data_path = "/Volumes/fflory/finreg/training/ift/mds/"
jsonl_data_path = "/Volumes/fflory/finreg/training/ift/jsonl/"


ift_train_df, ift_val_df = (
    get_spark().table("fflory.finreg.qa_dataset_2").randomSplit([0.99, 0.01])
)
ift_train_df.write.mode("overwrite").saveAsTable("fflory.finreg.qa_dataset_train")
ift_val_df.write.mode("overwrite").saveAsTable("fflory.finreg.qa_dataset_val")

# COMMAND ----------

from finreganalytics.utils import get_spark
from finreganalytics.dataprep import store_as_mds, store_as_jsonl
from finreganalytics.dataprep.ift_data_prep import prepare_ift_dataset

# COMMAND ----------

ift_completions_train_df = prepare_ift_dataset("fflory.finreg.qa_dataset_train", limit=-1)
ift_completions_val_df = prepare_ift_dataset("fflory.finreg.qa_dataset_val", limit=-1)

# COMMAND ----------

mds_data_path = "/Volumes/fflory/finreg/training/ift/mds/"
jsonl_data_path = "/Volumes/fflory/finreg/training/ift/jsonl/"


# COMMAND ----------

store_as_mds(ift_completions_train_df, os.path.join(mds_data_path, "train"))
store_as_jsonl(ift_completions_train_df, os.path.join(jsonl_data_path, "train.jsonl"))

store_as_mds(ift_completions_val_df, os.path.join(mds_data_path, "val"))
store_as_jsonl(ift_completions_val_df, os.path.join(jsonl_data_path, "val.jsonl"))
