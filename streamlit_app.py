import streamlit as st
from streamlit import session_state as ses
import pandas as pd
import numpy as np
import base64
from joblib import load

# Assuming the functions `predict_and_average` and `fill_na_from_lookup`
# are properly defined in the 'src.core' module
from src.core import predict_and_average, fill_na_from_lookup


# Custom CSS to enhance the UI
st.markdown("""
<style>
    .big-font {
        font-size:18px !important;
    }
    .title-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .container {
        padding: 10px;
    }
    .value-input {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


input_features = {
    'Waist Circumference (cm)': 'Окружность талии (см)',
    'Standing Height (cm)': 'Рост (см)',
    'Weight (kg)': 'Вес (кг)',
    'Systolic blood pressure average': 'Среднее артериальное давление (мм рт. ст.)',
    'Estimated Glomerular Filtration Rate (mL/min/1.73 m2)': 'Оценочная скорость клубочковой фильтрации (мл/мин/1.73 м²)',
    'alpha-tocopherol (µg/dL)': 'альфа-токоферол (мкг/дл)',
    'Serum homocysteine: SI (umol/L)': 'Сывороточный гомоцистеин (мкмоль/л)',
    'Serum ferritin (ng/mL)': 'Сывороточный ферритин (нг/мл)',
    'Serum creatinine (mg/dL)': 'Сывороточный креатинин (мг/дл)',
    'Serum blood urea nitrogen (mg/dL)': 'Сывороточный азот мочевины крови (мг/дл)',
    'Serum HDL cholesterol (mg/dL)': 'Сывороточный ХС ЛПВП (мг/дл)',
    'Serum albumin:  SI (g/L)': 'Сывороточный альбумин (г/л)',
    'Serum C-reactive protein (mg/dL)': 'С-реактивный белок сыворотки крови (мг/дл)'
}
# engineered features
#  'Body Mass Index (kg/m**2)',
#  'Waist to Height ratio'

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
    
    st.markdown("<div class='title-font'>Biological Age Calculator</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='big-font'>Биологический возраст - это количественный совокупный показатель вашего здоровья.</div>
""", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
<div class='big-font'>Пожалуйста, введите свои физиологические параметры, которые вы хотите включить в расчет биологического возраста. 
Вы можете ввести произвольное количество параметров. Наиболее важные параметры, которые более точно помогут сделать вычисления, находятся сверху.</div>
""", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Ваш пол', ('Мужской', 'Женский'), key='gender_select')
    with col2:
        age = st.number_input('Ваш хронологический возраст',  min_value=40, max_value=90, step=1)

    age_bins = np.arange(15, 90, 5)
    age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
    age_group = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]

    biomarker_container = st.container()
    biomarker_values = pd.DataFrame(index=[0]) # Initialize an empty DataFrame to collect biomarker values
    
    for feature_key, feature_name in input_features.items():

        # Initialize session state for button toggles if not already done
        if feature_key + '_use_avg' not in st.session_state:
            st.session_state[feature_key + '_use_avg'] = False

        biomarker_container.markdown(f'<span style="font-size: 30px;">{feature_name}</span>', unsafe_allow_html=True)

        use_avg = st.session_state[feature_key + '_use_avg']
        if not biomarker_container.checkbox(f'Использовать среднее', value=use_avg, key=feature_key + '_avg'):
            st.session_state[feature_key + '_use_avg'] = False
            # Let the user input their value
            user_value = biomarker_container.number_input(f'Ваше значение', key=feature_key)
            biomarker_values.loc[0, feature_key] = user_value
            # biomarker_container.write(f'Выбрано: {user_value}')
            biomarker_container.markdown(f'<span style="font-size: 25px;">input: {user_value}</span><br>---------------------------------------------------------------------------<br><br>', unsafe_allow_html=True)

        else:
            st.session_state[feature_key + '_use_avg'] = True
            # Get and display the average value
            avg_value = avgs_for_age_group.loc[age_group, feature_key].round(2)
            biomarker_values.loc[0, feature_key] = avg_value
            # biomarker_container.write(f'Выбрано: {avg_value}')
            biomarker_container.markdown(f'<span style="font-size: 25px;">input: {avg_value}</span><br>---------------------------------------------------------------------------<br><br>', unsafe_allow_html=True)

    # Предикт
    if st.button("Рассчитать"):

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
        st.write(f"Ваши значения: {list(zip(required_features, biomarker_values))}")
        
    st.divider()
    

    st.title("FAQ")
    # Folding text with info about the principle of biological age computation
    with st.expander("Принцип расчета биологического возраста Biological Age Computation."):
        st.markdown("""
                    тут будет описание
                    """)


if __name__ == "__main__":
    main()
