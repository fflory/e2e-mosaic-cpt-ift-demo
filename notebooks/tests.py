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

# import os
# from langchain_community.chat_models.databricks import ChatDatabricks
# from pyspark.sql.functions import rand

# from finreganalytics.dataprep.qagen import build_qa_eval_dataset
# from finreganalytics.utils import get_spark
# from finreganalytics.dataprep import store_as_mds, store_as_jsonl
# from finreganalytics.dataprep.ift_data_prep import (
#     prepare_ift_dataset,
#     loaf_huggingface_dataset,
# )

# try:
#     context = dbutils.entry_point.getDbutils().notebook().getContext()  # noqa
#     os.environ["DATABRICKS_HOST"] = context.apiToken().get()
#     os.environ["DATABRICKS_TOKEN"] = context.apiUrl().get()
# except:
#     pass

# COMMAND ----------

chunks_df = get_spark().read.table("fflory.finreg.splitted_documents").limit(2)
chunks = chunks_df.toPandas()["text"].values.tolist()
chunks

# COMMAND ----------

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", temperature=0.1)
EVALUATION_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination on Capital Requirements Regulation (CRR). The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.
Please generate exactly {num_questions_per_chunk} questions and no more. 
Do not include any further information.

Below is an example of a question.
Always format the output in JSON format as follows:

```json
[ 
"What problems addresses Capital Requirements Regulation?",
"What is Common Reporting Framework (COREP) ?" 
] 
``` """
QA_TEMPLATE_RAG = """
Context information is below.

---------------------
{context}
---------------------

You are an expert in European Financial Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
If the question is not related to one of these topics, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
Please do not repeat the answer and do not add any additional information. 

Question: {question}

Answer:
"""

# COMMAND ----------

qa_questions_df = build_qa_eval_dataset(
    chunks,
    llm_dbrx,
    question_prompt_template_str=EVALUATION_QUESTION_GENERATION_PROMPT_TMPL,
    answer_prompt_template_str=QA_TEMPLATE_RAG,
    num_questions_per_chunk=10,
)

display(qa_questions_df)  # noqa

# COMMAND ----------

import json
import re
from operator import itemgetter
from typing import Union, List
import pandas as pd

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

# COMMAND ----------

question_prompt_template_str=EVALUATION_QUESTION_GENERATION_PROMPT_TMPL
answer_prompt_template_str=QA_TEMPLATE_RAG
llm=llm_dbrx
# llm=ChatDatabricks(endpoint="ff_finreg_llama7b_ift", temperature=0.1)
num_questions_per_chunk=10
def parse(s: str, llm: BaseLanguageModel) -> Union[List[str], None]:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :param llm: LLM to fix syntax
    :return: parsed list of questions
    """
    try:
        arr = json.loads(extract_json_array(s))
        if arr:
            return [r.strip() for r in arr]
        else:
            return None
    except Exception as e:
        return None
      
def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\[.*]", s, re.DOTALL)
    if groups:
        return groups.group()
    else:
        return s

# COMMAND ----------

question_prompt = PromptTemplate(
        template=question_prompt_template_str,
        input_variables=["context", "num_questions_per_chunk"],
    )
answer_prompt = PromptTemplate(
    template=answer_prompt_template_str,
    input_variables=["question", "context"],
)
questions_chain = RunnableParallel(
    context=RunnablePassthrough(),
    num_questions_per_chunk=RunnableLambda(lambda x: num_questions_per_chunk),
) | RunnableParallel(
    context=itemgetter("context"),
    question=question_prompt | llm | StrOutputParser(),
).with_retry(
    stop_after_attempt=100, wait_exponential_jitter=False
)

questions_results = questions_chain.batch(chunks, config={"max_concurrency": 4})

questions_df = pd.DataFrame(
    [
        {
            "context": entry["context"].strip(),
            "question": parse(entry["question"], llm=llm),
        }
        for entry in questions_results
    ]
)
questions_df = questions_df.explode("question")
questions_dict_list = questions_df.to_dict(orient="records")
answers_chain = RunnableParallel(
    context=itemgetter("context"),
    question=itemgetter("question"),
    answer=answer_prompt | llm | StrOutputParser(),
).with_retry(stop_after_attempt=100)
answers_results = answers_chain.batch(
    questions_dict_list, config={"max_concurrency": 4}
)
res_df = pd.DataFrame(answers_results).dropna()

# COMMAND ----------

res_df
