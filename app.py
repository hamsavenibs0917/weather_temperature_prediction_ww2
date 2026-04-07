import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Weather App", layout="wide")
st.title("Weather Temperature Prediction ww2")


@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")
    df = df.drop(["Formatted Date", "Daily Summary", "Loud Cover"], axis=1, errors="ignore")

    if "Precip Type" in df.columns:
        mode_value = df["Precip Type"].mode()
        if not mode_value.empty:
            df["Precip Type"] = df["Precip Type"].fillna(mode_value[0])

    df = df.drop_duplicates()
    return df


@st.cache_resource
def train_model(df):
    model_df = pd.get_dummies(df, drop_first=True)
    X = model_df.drop(["Temperature (C)", "Apparent Temperature (C)"], axis=1, errors="ignore")
    y = model_df["Temperature (C)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X.columns.tolist(), y_test, y_pred


df = load_and_prepare_data()
model, feature_columns, y_test, y_pred = train_model(df)

st.subheader("1) Data Preview")
st.write("Dataset shape:", df.shape)
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2) Model Result")
st.write("R2 Score:", round(r2_score(y_test, y_pred), 4))
st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 4))

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_title("Actual vs Predicted Temperature")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

st.subheader("3) Predict New Temperature")
st.write("Enter values and click Predict")

input_df = df.drop(["Temperature (C)", "Apparent Temperature (C)"], axis=1, errors="ignore").copy()
user_input = {}

for col in input_df.columns:
    if is_numeric_dtype(input_df[col]):
        values = pd.to_numeric(input_df[col], errors="coerce").dropna()
        min_val = float(values.min()) if not values.empty else 0.0
        max_val = float(values.max()) if not values.empty else 1.0
        default_val = float(values.median()) if not values.empty else 0.0
        if min_val == max_val:
            max_val = min_val + 1.0
        user_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default_val)
    else:
        options = sorted(input_df[col].dropna().astype(str).unique().tolist())
        user_input[col] = st.selectbox(col, options)

if st.button("Predict"):
    user_row = pd.DataFrame([user_input])
    user_row = pd.get_dummies(user_row, drop_first=True)
    user_row = user_row.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(user_row)[0]
    st.success(f"Predicted Temperature (C): {prediction:.2f}")