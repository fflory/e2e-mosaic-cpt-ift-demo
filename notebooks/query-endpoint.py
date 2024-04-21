# Databricks notebook source
prompt = """[INST]Summarize the following article in a CFO-style, executive summary format. Include the Overview, Key Takeaways, Key Figures, and Key Action Items in your response and nothing else.[/INST]

ARTICLE:"""

# COMMAND ----------

article = """
  Generative AI is maturing swiftly, which may explain why many CFOs are evaluating how they might integrate the deep learning technology into their workflows. Indeed, generative AI may ultimately augment the finance function’s value proposition to the business, given the range of prospective use cases, from automating heavily manual processes to powering real-time predictive models.

CFOs might feel overwhelmed about where to start. As they implement strategic plans, keeping a close eye on inflation and the risk of recession, many may also wonder how they can begin to engage the technology to drive efficient growth. In Deloitte’s third quarter 2023 CFO Signals™ survey, 42% of the 116 respondents said their organizations are experimenting with generative AI, while another 15% said their companies are incorporating it into their business strategy.

Intelligent Design
Generative AI’s emergence represents the culmination of two technological breakthroughs. One is large language models (LLMs), algorithms that can analyze massive amounts of text. The other: machines with mammoth computing power. While LLMs lack an innate understanding of words, they can predict the next most likely word in a sentence based on patterns they detect— misfiring some of the time.

Generative AI and finance seem a natural fit, given that the former is powered by data and the latter draws upon abundant amounts of it. Deloitte’s third-quarter 2023 CFO Signals survey  found that two-thirds of surveyed CFOs are either experimenting with generative AI or reading and talking about it, while another 15% said they are incorporating it into strategy.

Integrating generative AI into a complicated workflow can be—no surprise—complex. Just reaching the testing phase can feel like an accomplishment. In preparation, CFOs can take some proactive steps, including the following:
Get up to speed. Finance leaders don’t need to be experts, but they should know enough to decide on outcomes, such as opening up new revenue streams. And they should understand how generative AI uses data to make decisions and help demystify the process for team members.
Collaborate with other functional leaders. In the third-quarter 2023 CFO Signals survey, 59% of respondents said that ownership of generative AI within their organization belongs to the CTO, CIO, Chief Data Officer, or equivalent. However, CFOs and other leaders should not perceive generative AI as merely an issue for IT to resolve. Instead, they should participate in shaping strategy and driving execution, given that generative AI can represent a new way of reaching decisions.
Assess data infrastructure and related needs.  CFOs need to work closely with their chief data officers, sharing their use cases with them and ensuring that all necessary data, both internal and external, is appropriately structured and available. As tempting as it may be to roll out a pilot project, unleashing the capabilities of generative AI at full throttle can trigger risks, only some of which—cybersecurity, data privacy, and regulatory compliance—are knowable in advance.
The technology requires vast amounts of data and serious processing power. Given the investment needed, consider starting with a small, targeted set of use cases aimed at delivering specific benefits. For CFOs, that can mean identifying a challenge where the company has already been collecting data and has established frameworks to govern the process.

Real-World AI
The move to real-world applications may mean first deploying generative AI to improve efficiency and productivity. In fact, in the third quarter 2023 CFO Signals survey, 52% of respondents ranked “reduce costs” among the top three benefits they hoped to achieve by using the technology; a slightly smaller proportion, 45%, prioritized increasing margins, efficiencies, and/or productivity.For finance, that can mean starting by using the tool to generate scripts for earnings calls or to anticipate analyst questions. It might also be used to produce succinct summaries of meeting minutes or create a coherent impact analysis regarding new regulations.

Cutting manual tasks may save money, but it doesn’t typically confer competitive advantage.
However, integrating generative AI into the organization’s processes and workflows—which will need to be digitized to efficiently adopt the technology—could increase gross revenue. For example, the technology may enable businesses to bring products to market faster. By analyzing data from a variety of sources, it might also help shape the design of new offerings, ultimately generating insights that can help leaders better understand the business.  CFOs should also consider some of the following actions:
*Assessing data readiness. Generative AI’s appetite for structured and unstructured data drawn from a variety of sources can be properly fed only when that data is put into a consistent format and centralized to safeguard consistency. Meeting regulatory requirements should also be a priority, especially if consumer data is involved.
*Bridging skills gaps. Prior to implementing generative AI, CFOs may need to determine whether the business has sufficient numbers of tech-savvy finance professionals. In the third-quarter 2023 CFO Signals survey, 63% of respondents cited “talent resources and capabilities” as one of their top three barriers to adopting and deploying the technology.
*Identify quick wins. To generate ROI, a simple use case may be to set up a chatbot to answer basic HR-related questions about travel or vacation policies. Such implementations need to be part of a long-term strategy; without an overarching view, the effort can result in a loose federation of use cases.
*Clarify the role of humans. CFOs and other leaders may want to emphasize that the use of generative AI is intended to enhance employees’ experience and productivity when introducing the technology. Generative AI’s capabilities should be seen as supplementing cognitive abilities of the organization’s employees—not replacing them.

Employees should be regarded as customers of generative AI, not rivals. Granted, it’s hard to predict what the future will bring. But the real value of generative AI for finance may be in supporting decision-making—not taking responsibility for doing it.

—by James Glover and Ranjit Rao, principals, Finance and Performance; Gina Schaefer, managing director, AI & Data; and Kate Schmidt, chief operating officer, AI Strategic Growth Offering, all Deloitte Consulting LLP; and Court Watson, partner, Controllership Transformation, Deloitte & Touche LLP.
"""

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
        "prompt": prompt + article,
        "temperature": 0.1,
        "max_tokens": 500,
        "n": 1,
    },
)

# COMMAND ----------

chat_response

# COMMAND ----------

import pprint
summarization = """
Overview:
Generative AI is maturing swiftly, and many CFOs are evaluating how they might integrate the deep learning technology into their workflows. The technology can potentially augment the finance function’s value proposition to the business, given the range of prospective use cases, from automating heavily manual processes to powering real-time predictive models.

Key Takeaways:
* 42% of CFOs surveyed by Deloitte are experimenting with generative AI, while another 15% said their companies are incorporating it into business strategy.
* Generative AI uses large language models (LLMs), algorithms that can analyze massive amounts of text, and machines with mammoth computing power.
* CFOs should get up to speed on generative AI, collaborate with other functional leaders, assess data infrastructure and related needs, and identify quick wins.

Key Stats:\
  * 63% of surveyed CFOs cited “talent resources and capabilities” as one of their top three barriers to adopting and deploying generative AI.
  * 52% of surveyed CFOs ranked “reduce costs” among the top three benefits they hoped to achieve by using generative AI.
  * 45% of surveyed CFOs prioritized increasing margins, efficiencies, and/or productivity as a benefit of using generative AI.
  
Key Action Items:
* CFOs should assess data readiness, bridge skills gaps, identify quick wins, and clarify the role of humans when introducing generative AI into the organization.
* Finance leaders should participate in shaping strategy and driving execution for generative AI, given that the technology can represent a new way of reaching decisions.
* CFOs should work closely with their chief data officers, sharing use cases with them and ensuring that all necessary data, both internal and external, is appropriately structured and available.'
"""

# COMMAND ----------

pprint.pp(summarization)

# COMMAND ----------


