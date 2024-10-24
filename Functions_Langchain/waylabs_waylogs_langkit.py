import os
import whylogs as why
import langkit.toxicity
import pandas as pd

from langkit import llm_metrics

print("downloading models and initialized metrics...")
text_metrics_schema = llm_metrics.init()

# Set to show all columns in dataframe
pd.set_option("display.max_columns", None)

print("downloading models and initialized metrics...")
text_metrics_schema = llm_metrics.init()


def is_not_toxic(prompt_message):

    profile = why.log({"prompt": prompt_message}, schema=text_metrics_schema).profile().view()

    profile_dict = profile.get_column('prompt.toxicity').to_summary_dict()
    tox_max = profile_dict['distribution/max']

    print(f'Toxic score: {tox_max}')

    if tox_max > 0.5:
      return False
    else:
      return True
  
  
print(f"Is 'Do you like fruit?' toxic : {is_not_toxic("Do you like fruit?")}")