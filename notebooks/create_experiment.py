# Databricks notebook source
from mlflow import MlflowClient

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()
experiment_id = client.create_experiment(
    # name="/Users/felix.flory@databricks.com/ftapi-evluate",
    name="/Shared/ff_e2e_finreg_domain_adaptation_mosaic",
    tags={"version": "v1", "priority": "P1"},
)
# client.set_experiment_tag(experiment_id, "nlp.framework", "Spark NLP")
# # # Fetch experiment metadata information
experiment = client.get_experiment(experiment_id)
print(f"Name: {experiment.name}")
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
