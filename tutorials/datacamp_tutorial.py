"""Causal Inference with DoWhy tutorial from DataCamp.

https://www.datacamp.com/tutorial/intro-to-causal-ai-using-the-dowhy-library-in-python
Another blog from DataCamp:
https://www.datacamp.com/blog/what-is-causal-ai
"""

# %%
import numpy as np
import pandas as pd
import dowhy

from dowhy import CausalModel
import dowhy.datasets

# Set seed to enable exact replication
np.random.seed(1)

# %%
# Simulate sample data
data = dowhy.datasets.linear_dataset(
    beta=1,
    num_common_causes=2,
    num_discrete_common_causes=1,
    num_instruments=1,
    num_samples=10000,
    treatment_is_binary=True,
)

df = data["df"]
# Treatment variable: WFH
# Outcome variable: Productivity
# Common causes
# W0: Introversion (Continuous)
# W1: Number of children (Categorical)
# Insrumental variable: Z0 (Binary)
# %%
data["dot_graph"]
# %%
# Linear Regresion
# Run a linear regression of column y on v0 in df
import statsmodels.api as sm

X = df["v0"].astype(float)
y = df["y"].astype(float)

X = sm.add_constant(X)

ols = sm.OLS(y, X).fit()

# Display a more parsimonious results summary
print(ols.summary().tables[1])
# %%
# Using a causal model
# Check whether causal effect is identified and return target estimands
# Create a causal model from the data and given graph.
model = CausalModel(
    data=df,
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"],
)
# Check whether causal effect is identified and return target estimands
identified_estimand = model.identify_effect()

# %%
# Estimate the causal effect using inverse probability weighting
estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.propensity_score_weighting"
)
estimate.value

# %%
# Check sensitivity of obtained estimate to unobserved confounders
refute_results = model.refute_estimate(
    identified_estimand, estimate, method_name="add_unobserved_common_cause"
)

# %%
# Using intrumental variable
# Check whether causal effect is identified and return target estimands
# It doesn't rely on assumption about the number of unobserved common causes of WFH and Productivity
# Estimates less precise but more robust due to weaker assumptions
iv_estimate = model.estimate_effect(
    identified_estimand, method_name="iv.instrumental_variable"
)
iv_estimate.value
# %%
