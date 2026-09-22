# The Judge — Bias & Fairness in AI

An end-to-end machine learning project auditing algorithmic bias in an income prediction model.

## Overview
This project builds a Random Forest classifier (86.4% accuracy) to predict whether an individual
earns above or below $50K/year, using the UCI Adult Census Income dataset. Sex and race were
deliberately excluded from training to test whether that alone makes the model fair.

**Finding:** it doesn't. The model still shows significant prediction disparities across sex and
race groups — driven by proxy variables like marital status and relationship category, which are
strongly correlated with the excluded attributes.

## What's in this repo
- `The_Judge_Bias_Fairness.ipynb` — full data cleaning, modeling, and fairness audit
- `app.py` — Streamlit app for live income predictions with model-reasoning explanations
- `The_Judge_Deck.pptx` — presentation slides summarizing findings

## Tools used
Python, pandas, scikit-learn, Streamlit, Jupyter Notebook

## Key result
A Random Forest model excluding sex/race from training still reproduced real-world income
disparities — proving that "fairness through unawareness" does not guarantee a fair outcome.
