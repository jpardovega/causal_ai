import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import dowhy.datasets

# Set seed for reproducibility
np.random.seed(123)

# Generate synthetic data
n_samples = 10000

# Create base synthetic data with continuous common causes
data = dowhy.datasets.linear_dataset(
    beta=5,  # Effect size of price change on retention
    num_common_causes=2,  # For age and tenure
    num_instruments=0,
    num_samples=n_samples,
    treatment_is_binary=False,  # price_change is continuous
)

df = data["df"]

# Rename columns to match our use case
df = df.rename(
    columns={"v0": "price_change", "y": "retention", "w0": "age", "w1": "tenure"}
)

# Add categorical policy_type
df["policy_type"] = np.random.choice(
    ["basic", "standard", "premium"], size=n_samples, p=[0.3, 0.5, 0.2]
)

# Add claims_count following a Poisson distribution
df["claims_count"] = np.random.poisson(lam=1.5, size=n_samples)

# Adjust retention to be binary (0 or 1) based on a probability
prob_retention = 1 / (1 + np.exp(-df["retention"]))  # Sigmoid transformation
df["retention"] = (prob_retention > 0.5).astype(int)

# Scale price_change to be more realistic (-20% to +20%)
df["price_change"] = df["price_change"] * 10  # Scale to percentage changes

# Adjust age to be more realistic (25 to 75 years)
df["age"] = (
    25 + (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min()) * 50
)

# Adjust tenure to be more realistic (0 to 20 years)
df["tenure"] = (
    (df["tenure"] - df["tenure"].min()) / (df["tenure"].max() - df["tenure"].min()) * 20
)

print("\nSynthetic Dataset Summary:")
print(df.describe())
print("\nPolicy Type Distribution:")
print(df["policy_type"].value_counts(normalize=True))

model = CausalModel(
    data=df,
    treatment="price_change",
    outcome="retention",
    common_causes=["age", "tenure", "policy_type", "claims_count"],
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.propensity_score_matching"
)

model.refute_estimate(
    identified_estimand, estimate, method_name="placebo_treatment_refuter"
)

from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

est = LinearDML(
    model_y=RandomForestRegressor(), model_t=LassoCV(), discrete_treatment=False
)

est.fit(
    Y=df["retention"],
    T=df["price_change"],
    X=df[["age", "tenure", "policy_type", "claims_count"]],
)
treatment_effects = est.effect(df[["age", "tenure", "policy_type", "claims_count"]])

import matplotlib.pyplot as plt

plt.hist(treatment_effects, bins=30)
plt.title("Distribution of Estimated Price Elasticity")
plt.xlabel("Effect of Price Change on Retention")
plt.ylabel("Customer Count")
plt.show()
