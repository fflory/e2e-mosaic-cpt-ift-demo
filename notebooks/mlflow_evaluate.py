# Databricks notebook source
# MAGIC %md
# MAGIC see https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# MAGIC
# MAGIC ML runtime 15 has most packages except textstat

# COMMAND ----------

# MAGIC %pip install textstat # evaluate torch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F

# COMMAND ----------

df_qa_with_preds = spark.table("fflory.finreg.qa_dataset_output").withColumn("inputs", F.col("question"))

# COMMAND ----------

display(df_qa_with_preds.limit(4))

# COMMAND ----------

endpoint_name = "databricks-dbrx-instruct"

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

evaluation_data = (
    df_qa_with_preds
    .limit(4)
    .toPandas()
)

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = evaluation_data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="generated_answer", # prediction column_name from eval_df
                                   targets = "answer",
                                   extra_metrics=[answer_correctness_metrics, professionalism] # 
                                   )
    
eval_results.metrics

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

df_genai_metrics.columns

# COMMAND ----------

df_genai_metrics.loc[df_genai_metrics['toxicity/v1/score']*100>15]

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")
