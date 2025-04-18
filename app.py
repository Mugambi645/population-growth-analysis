# POPULATION PREDICTION STREAMLIT APP

# IMPORTING LIBRARIES
import pandas as pd 
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from numerize import numerize
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# LOAD DATASETS
countries_pop = pd.read_csv('datasets/Countries_Population_final.csv')
countries_name = pd.read_csv('datasets/Countries_names.csv')

# PAGE TITLE
col1, col2, col3 = st.columns([2, 6, 2])
with col2:
    st.info('# :blue[Analysing Population Growth]')

# HELPER FUNCTION: MODEL PREDICTION
@st.cache_data
def create_polynomial_regression_model(country: str, degree: int, year: int):
    X = countries_pop['Year']
    y = countries_pop[country]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    y_test_predict = poly_model.predict(poly_features.transform(X_test))
    prediction = poly_model.predict(poly_features.transform([[year]]))
    r2_test = r2_score(Y_test, y_test_predict)

    return prediction[0], int(r2_test * 100)

# MAIN DASHBOARD
col1, col2 = st.columns(2)

with col1:
    # USER INPUT
    country = st.selectbox('PLEASE SELECT ANY COUNTRY', sorted(countries_name['Country_Name']))
    default_year = str(countries_pop['Year'].max() + 5)
    year_input = st.text_input('PLEASE ENTER YEAR', default_year)

    if year_input.isnumeric():
        year = int(year_input)
        if year < countries_pop['Year'].max():
            st.warning("Please enter a year beyond the latest available data.")
        else:
            predicted_pop, accuracy = create_polynomial_regression_model(country, 2, year)
            pred_formatted = numerize.numerize(predicted_pop)

            # OUTPUT SECTION
            st.write("#### :green[ALGORITHM: ] POLYNOMIAL REGRESSION")
            st.write(f"#### :green[ACCURACY: ] {accuracy}%")
            st.write(f"#### :green[COUNTRY: ] {country.upper()}")
            st.write(f"#### :green[YEAR: ] {year}")
            st.write(f"#### :green[PREDICTED POPULATION: ] {pred_formatted}")

            # DOWNLOAD OPTION
            output_csv = pd.DataFrame({
                'Country': [country],
                'Year': [year],
                'Predicted Population': [int(predicted_pop)]
            })
            csv = output_csv.to_csv(index=False)
            st.download_button("📥 Download Prediction", data=csv, file_name="population_prediction.csv", mime='text/csv')
    else:
        st.warning("Please enter a valid numeric year.")

# CHART SECTION
with col2:
    if year_input.isnumeric() and int(year_input) >= countries_pop['Year'].max():
        st.write(f'#### :green[{country.upper()}\'S POPULATION TREND]')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=countries_pop['Year'], y=countries_pop[country],
            name="Historical Data",
            line=dict(color='green', width=4)
        ))
        fig.add_trace(go.Scatter(
            x=[int(year_input)], y=[predicted_pop],
            name='Predicted ' + year_input,
            mode='markers',
            marker_symbol='star',
            marker=dict(size=20, color='red')
        ))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The above plot shows the population trend from 1960 to 2021. The star represents the predicted population for the selected year.")
