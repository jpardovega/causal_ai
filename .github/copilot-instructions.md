# Copilot Instructions for `causal_ai`

## Project Overview
- **Purpose:** Experimental playground for causal inference methods in data science, with a focus on hands-on tutorials and reproducible experiments.
- **Main Example:** See `tutorials/datacamp_tutorial.py` for a guided workflow using the DoWhy library and synthetic data.

## Key Patterns & Conventions
- **Tutorials:** Place all tutorial scripts in the `tutorials/` directory. Scripts are typically self-contained, with clear data simulation, modeling, and result sections.
- **Reproducibility:** Set random seeds (e.g., `np.random.seed(1)`) at the start of scripts to ensure results are replicable.
- **Data Simulation:** Use `dowhy.datasets.linear_dataset` for generating synthetic datasets with specified causal structures.
- **Causal Modeling:** Use `dowhy.CausalModel` for defining and analyzing causal graphs. Document treatment, outcome, common causes, and instruments in comments.
- **Statistical Analysis:** Use `statsmodels` for regression and statistical summaries. Print concise tables for results.
- **Documentation:** Reference external tutorials or blogs in script docstrings for context and reproducibility.

## Developer Workflows
- **Dependencies:**
  - Required: `numpy`, `pandas`, `dowhy`, `statsmodels`
  - Install with: `pip install numpy pandas dowhy statsmodels`
- **Running Tutorials:**
  - Execute scripts in `tutorials/` directly (e.g., `python tutorials/datacamp_tutorial.py`).
  - Jupyter or VS Code interactive cells (`# %%`) are supported for stepwise execution.
- **Adding New Tutorials:**
  - Follow the structure of `datacamp_tutorial.py`: docstring with references, seed setting, data simulation, modeling, and result output.
  - Clearly comment on variable roles (treatment, outcome, common causes, instruments).

## Integration & Extensibility
- **External Resources:** Scripts may link to external tutorials or blogs for reproducibility and learning.
- **No build system or test suite** is present; focus is on experimentation and documentation.

## Example: Causal Model Setup
```python
model = CausalModel(
    data=df,
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["dot_graph"]
)
```

## Contact
- For questions, see script docstrings or README references.
