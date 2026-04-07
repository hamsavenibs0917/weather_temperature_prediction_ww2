# Weather Temperature Prediction Capstone

This project predicts temperature from historical weather data using machine learning.
It includes:
- A Jupyter notebook workflow for data cleaning, training, evaluation, and manual prediction
- A Streamlit web app for interactive prediction

## Project Files
- app.py: Streamlit app
- capstone_project.ipynb: End-to-end notebook workflow
- weatherHistory.csv: Dataset
- requirements.txt: Python dependencies

## Setup
1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

After running, open the local URL shown in the terminal (usually http://localhost:8501).

## Run the Notebook
1. Open capstone_project.ipynb in VS Code or Jupyter
2. Run cells from top to bottom
3. Enter feature values when prompted in the final prediction cell

## Model Overview
- Linear Regression (with scaling)
- Random Forest Regressor
- Evaluation metrics: R2 score and MAE

## Notes
- The notebook drops non-useful columns and handles missing values in Precip Type.
- The Streamlit app trains a Random Forest model and allows interactive predictions.
