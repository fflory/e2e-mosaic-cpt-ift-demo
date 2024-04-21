# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

import os.path
import mcli

from finreganalytics.utils import setup_logging, get_dbutils

setup_logging()

SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b",
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-8k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
]
get_dbutils().widgets.combobox(
    "base_model", "mosaicml/mpt-7b-8k", SUPPORTED_INPUT_MODELS, "base_model"
)
get_dbutils().widgets.text(
    "data_path", "/Volumes/fflory/finreg/training/cpt/text/train/", "data_path" 
)

get_dbutils().widgets.text("training_duration", "5ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "5e-7", "learning_rate")

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
data_path = get_dbutils().widgets.get("data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="felix-flory", key="mosaic-token"))


# COMMAND ----------

from mcli import RunStatus

run = mcli.create_finetuning_run(
    model=base_model,
    train_data_path=f"dbfs:{data_path}",
    #eval_data_path=f"dbfs:{data_path}",
    save_folder="dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/",
    task_type="CONTINUED_PRETRAIN",
    training_duration=training_duration,
    learning_rate=learning_rate,
    experiment_tracker={
        "mlflow": {
            "experiment_path": "/Shared/ff_e2e_finreg_domain_adaptation_mosaic",
            "model_registry_path": "fflory.finreg.crr_llama2_cpt_v1",
        }
    },
    disable_credentials_check=True,
)
print(f"Started Run {run.name}. The run is in status {run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(run.name):
    print(s)
