import streamlit as st
from streamlit import session_state as ses
import pandas as pd
import numpy as np
import base64
from joblib import load

# Assuming the functions `predict_and_average` and `fill_na_from_lookup`
# are properly defined in the 'src.core' module
from src.core import predict_and_average, fill_na_from_lookup

input_features = [
 'Waist Circumference (cm)',
 'Standing Height (cm)',
 'Weight (kg)',
 'Systolic blood pressure average',
 'Estimated Glomerular Filtration Rate (mL/min/1.73 m2)',
 'alpha-tocopherol (µg/dL)',
 'Serum homocysteine: SI (umol/L)',
 'Serum ferritin (ng/mL)',
 'Serum creatinine (mg/dL)',
 'Serum blood urea nitrogen (mg/dL)',
 'Serum HDL cholesterol (mg/dL)',
 'Serum albumin:  SI (g/L)',
 'Serum C-reactive protein (mg/dL)'
 
#  'Body Mass Index (kg/m**2)',
#  'Waist to Height ratio'
]

models_paths = ['models/catboost/model_fold_1.joblib',
 'models/catboost/model_fold_2.joblib',
 'models/catboost/model_fold_3.joblib',
 'models/catboost/model_fold_4.joblib',
 'models/catboost/model_fold_5.joblib']

avgs_for_age_group = pd.read_csv('avgs_for_age_group.csv', index_col='Age Group')  # Assuming 'Age Group' is set as index

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" width="150" height="150"/>'
    st.write(html, unsafe_allow_html=True)
    
def main():
    # Load and render the logo
    logo_path = "front/logo.svg"
    with open(logo_path, "r") as file:
        logo_svg = file.read()
    render_svg(logo_svg)  
    
    st.title("Biological Age Calculator")
    st.write("""
Биологический возраст - это количествый совокупный показатель вашего здоровья
""")
    st.divider()


    st.write("""
Пожалуйста, введите свои физиологические параметры, которые вы хотите включить в расчет биологического возраста. 
Вы можете ввести произвольное количество параметров. Наиболее важные параметры которые более точно помогут сделать вычисления находятся сверху. 

Мы рекомендуем обязательно использовать первые 5 параметров для более точного определения биовозраста.
Однако возможны любые комбинации. 
""")
    st.divider()
    
    gender = st.selectbox('Ваш пол', ('Мужской', 'Женский'), key='gender_select')
    age = st.number_input('Ваш хронологический возраст',  min_value=40, max_value=90, step=1)
    
    age_bins = np.arange(15, 90, 5)
    age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
    age_group = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]

    # Сбор данных
    biomarker_container = st.container()
    for feature_name in input_features:
        feature_key = feature_name

        # Initialize session state for button toggles if not already done
        if feature_key + '_use_avg' not in st.session_state:
            st.session_state[feature_key + '_use_avg'] = False

        use_avg = st.session_state[feature_key + '_use_avg']
        
        if biomarker_container.checkbox(f'Use average for {feature_name}', value=use_avg, key=feature_key + '_avg'):
            st.session_state[feature_key + '_use_avg'] = True
            # Get and display the average value
            avg_value = avgs_for_age_group.loc[age_group, feature_name].round(2)
            biomarker_container.write(f'Average value for {feature_name} in : {avg_value}')
        else:
            st.session_state[feature_key + '_use_avg'] = False
            # Let the user input their value
            user_value = biomarker_container.number_input(f'Value for {feature_name}', key=feature_key)

    # Предикт
    if st.button("Рассчитать"):
        # Initialize an empty DataFrame to collect biomarker values
        biomarker_values = pd.DataFrame(index=[0])
        for feature_name in input_features:
            feature_key = feature_name
            # Check if the user opted to use the average value for the current feature
            if st.session_state.get(feature_key + '_use_avg', False):
                avg_value = avgs_for_age_group.loc[age_group, feature_name]
                biomarker_values.loc[0, feature_name] = avg_value
            else:
                user_value = st.session_state.get(feature_key, np.nan)
                biomarker_values.loc[0, feature_name] = user_value

        # FEATURE ENGINEERING
        biomarker_values['gender'] = 1 if gender == 'Мужской' else 2 
        biomarker_values['Body Mass Index (kg/m**2)'] = (biomarker_values['Weight (kg)'] / ((biomarker_values['Standing Height (cm)']*0.01)**2)).round(2)
        biomarker_values['Waist to Height ratio'] = biomarker_values['Waist Circumference (cm)'] / biomarker_values['Standing Height (cm)']


        model = load(models_paths[0])  # Load one model to get the feature names
        required_features = model.feature_names_
        biomarker_values = biomarker_values.reindex(columns=required_features)

        # Handle possible missing columns due to conditional feature input
        for missing_col in set(required_features) - set(biomarker_values.columns):
            biomarker_values[missing_col] = np.nan  # Fill missing columns with NaNs

        # Predict the biological age using the provided biomarker values
        ba = predict_and_average(biomarker_values, models_paths)

        # Display the estimated biological age
        st.write(f"Ваш биологический возраст: {int(ba)}")
        
    st.divider()
    

    st.title("FAQ")
    # Folding text with info about the principle of biological age computation
    with st.expander("Принцип расчета биологического возраста Biological Age Computation."):
        st.markdown("""
                    тут будет описание
                    """)


if __name__ == "__main__":
    main()
