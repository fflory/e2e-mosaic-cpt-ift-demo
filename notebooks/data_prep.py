# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# dbutils.fs.cp("/Volumes/msh/finreg/training", "/Volumes/fflory/finreg/training", True)

# COMMAND ----------

# dbutils.fs.cp("/Volumes/msh/finreg/data", "/Volumes/fflory/finreg/data", True)

# COMMAND ----------

from finreganalytics.dataprep.dataloading import load_and_clean_data, split
from finreganalytics.utils import get_spark

data_path = "/Volumes/fflory/finreg"

# COMMAND ----------

doc_df = load_and_clean_data(f"{data_path}/data")
display(doc_df)

# COMMAND ----------

splitted_df = split(
    doc_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df)

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable("fflory.finreg.splitted_documents")


# COMMAND ----------

cpt_df = get_spark().read.table("fflory.finreg.splitted_documents")
cpt_train_df, cpt_val_df = cpt_df.select("text").randomSplit([0.98, 0.02])
cpt_train_df.write.mode("overwrite").format("text").save(
    f"{data_path}/training/cpt/text/train"
)
cpt_val_df.write.mode("overwrite").format("text").save(
    f"{data_path}/training/cpt/text/val"
)

# COMMAND ----------

# MAGIC %sql select count(1) from text.`/Volumes/fflory/finreg/training/cpt/text/train`

# COMMAND ----------

# MAGIC %sql select count(1) from text.`/Volumes/fflory/finreg/training/cpt/text/val`

# COMMAND ----------


!rm /Volumes/fflory/finreg/training/cpt/text/val/_committed_*
!rm /Volumes/fflory/finreg/training/cpt/text/val/_started_*
!rm /Volumes/fflory/finreg/training/cpt/text/val/_SUCCESS

# COMMAND ----------

!rm /Volumes/fflory/finreg/training/cpt/text/train/_committed_*
!rm /Volumes/fflory/finreg/training/cpt/text/train/_started_*
!rm /Volumes/fflory/finreg/training/cpt/text/train/_SUCCESS
