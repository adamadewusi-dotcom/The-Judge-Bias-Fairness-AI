import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="The Judge — Income Predictor", page_icon="⚖️", layout="centered")
st.title("⚖️ The Judge")
st.subheader("Income Prediction Model")
st.write(
    "This app predicts whether someone earns more than $50K/year based on the "
    "Adult Census Income dataset. **Sex and race were excluded from model training** — "
    "our full fairness audit (in the notebook and slides) shows the model can still "
    "reflect real-world disparities through other correlated features."
)

# ---------- Load model and reference data ----------
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_columns():
    return pd.read_csv("X_train_columns.csv")["column"].tolist()

@st.cache_data
def load_feature_importance():
    return pd.read_csv("feature_importance.csv")

model = load_model()
model_columns = load_columns()
feature_importance = load_feature_importance()

FEATURE_LABELS = {
    "capital.gain": "Capital gain",
    "marital.status_Married-civ-spouse": "Being married (civilian spouse)",
    "education.num": "Education level",
    "age": "Age",
    "marital.status_Never-married": "Never having been married",
    "hours.per.week": "Hours worked per week",
    "capital.loss": "Capital loss",
    "relationship_Not-in-family": "Not living with family",
    "relationship_Own-child": "Being a dependent child in the household",
    "occupation_Exec-managerial": "Working in an executive/managerial role",
}

WORKCLASS = ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
MARITAL = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
OCCUPATION = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
              'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
              'Tech-support', 'Transport-moving']
RELATIONSHIP = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
COUNTRY = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'China', 'Other']

# ---------- Input form ----------
st.header("1. Enter Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 17, 90, 35)
    education_num = st.slider("Education level (1=lowest, 16=Doctorate)", 1, 16, 10)
    hours_per_week = st.slider("Hours worked per week", 1, 99, 40)
    workclass = st.selectbox("Workclass", WORKCLASS)
    occupation = st.selectbox("Occupation", OCCUPATION)

with col2:
    marital_status = st.selectbox("Marital status", MARITAL)
    relationship = st.selectbox("Relationship", RELATIONSHIP)
    native_country = st.selectbox("Native country", COUNTRY)
    if native_country == "Other":
        st.caption("\u2139\ufe0f \"Other\" is approximated using the model's baseline country category, since only the most common countries have their own dropdown option.")
    capital_gain = st.number_input("Capital gain ($)", min_value=0, max_value=99999, value=0, step=100)
    capital_loss = st.number_input("Capital loss ($)", min_value=0, max_value=4356, value=0, step=50)

# ---------- Build feature vector ----------
def build_input_row():
    row = {col: 0 for col in model_columns}
    row['age'] = age
    row['education.num'] = education_num
    row['capital.gain'] = capital_gain
    row['capital.loss'] = capital_loss
    row['hours.per.week'] = hours_per_week

    wc_col = f"workclass_{workclass}"
    if wc_col in row:
        row[wc_col] = 1

    ms_col = f"marital.status_{marital_status}"
    if ms_col in row:
        row[ms_col] = 1

    occ_col = f"occupation_{occupation}"
    if occ_col in row:
        row[occ_col] = 1

    rel_col = f"relationship_{relationship}"
    if rel_col in row:
        row[rel_col] = 1

    if native_country != "Other":
        nc_col = f"native.country_{native_country}"
        if nc_col in row:
            row[nc_col] = 1

    return pd.DataFrame([row])[model_columns]

# ---------- Predict ----------
st.divider()
if st.button("Predict Income Bracket", type="primary"):
    X_input = build_input_row()
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    st.header("2. Result")
    if pred == 1:
        st.success(f"### Predicted: **>$50K/year**")
    else:
        st.info(f"### Predicted: **≤$50K/year**")

    st.write(f"Model confidence: **{max(proba)*100:.1f}%**")
    st.progress(float(max(proba)))

    st.divider()
    st.header("3. Why This Kind of Result?")
    st.write(
        "The model doesn't explain individual predictions directly, but we can show which factors "
        "it relies on most overall, based on what it learned across the whole training set:"
    )
    top5 = feature_importance.head(5).copy()
    top5["label"] = top5["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    st.bar_chart(top5.set_index("label")["importance"])
    st.caption(
        "Capital gain and marital status are the two strongest drivers of the model's predictions — "
        "together they matter more than age, hours worked, or occupation."
    )
