import streamlit as st
from streamlit import session_state as ses
import pandas as pd
import base64
from joblib import load

def predict_and_average(X, models_saved):
    predictions = 0
    
    for model_filename in models_saved:
        model = load(model_filename)
        pred = model.predict(X)
        predictions += pred
    
    predictions /= len(models_saved)
    
    return predictions

use_cols = ['seqn', 'bup', 'tcp', 'crp', 'cep', 'fvc', 'amp', 'wbc', 'sbp', 'fev', 'gender']

feature_include = {
    'fev':'Forced expiratory volume (mL)', #~2000 руб в москве
    'fvc':'Forced Vital Capacity', # или этот вместо fev
    'sbp':'Systolic blood pressure', # ---  
    'bup':'Serum blood urea nitrogen (mg/dL)', # 380 rub gemotest,
    'tcp':'Serum cholesterol (mg/dL)', # 255 rub helix
    'cep':'Serum creatinine (mg/dL)', # 255 rub helix
    'amp':'Serum albumin (g/dL)', # 350 руб helix, 
    'wbc':'White blood cell count', # 650 руб invitro
    'crp':'Serum C-reactive protein (mg/dL)', # 510 rub helix
    'seqn': 'hz'
}
models_saved = ['models/model_fold_1.joblib',
 'models/model_fold_2.joblib',
 'models/model_fold_3.joblib',
 'models/model_fold_4.joblib',
 'models/model_fold_5.joblib']


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="150" height="150"/>' % b64
    st.write(html, unsafe_allow_html=True)

# Main Streamlit app
def main():
    # st.sidebar.subheader("About")
    # st.sidebar.write("""
    #                     With the help of a small number of easily accessible parameters of your body. 
    #                     Nevertheless, we tried to carefully select the parameters for the calculation 
    #                     so that they are as informative as possible.
    #                  """)

    f = open("front/logo.svg", "r")
    lines = f.readlines()
    line_string=''.join(lines)
    render_svg(line_string)  
    
    st.title("Biological Age Calculator")
    st.write("""
              Пожалуйста, введите свои физиологические параметры, которые вы хотите включить в расчет биологического возраста. Вы можете использовать любую комбинацию функций, включая или исключая любую из них.
              Признаки упорядочены по их значимости для расчета биологического возраста, так что первый из них имеет высшее значение. 
              Мы рекомендуем обязательно использовать первые 5 параметров для более точного определения биовозраста.
              Однако возможны любые комбинации. 
             """)
    st.divider()

    c1, c2 = st.columns([3, 3])
    with st.container():
        c1.write('') # for additional gap
        gender = c1.selectbox('Выберите ваш пол', ('Мужской', 'Женский'))

    #create input fields for biomarkers
    for bkey, bname in feature_include.items():
        c1, c2, c3 = st.columns([4, 3, 3])
        with st.container():
            usebio = c1.checkbox(bname, value=True)
            if usebio:
                user_button = c3.button(label="Use my value", key=bkey+'_user', use_container_width=True)
                ses[bkey] = 0. if bkey not in ses else ses[bkey]
                ses[bkey + '_button_state'] = False if bkey + '_button_state' not in ses else ses[bkey + '_button_state']

                # avg_button = c3.button(label="Use average value", key=bkey+'_avg', use_container_width=True)
                # if avg_button:
                    # ses[bkey] = predict_average_feature(model, bkey, age=age_value)
                    # ses[bkey + '_button_state']=True
                
                if user_button:
                    ses[bkey + '_button_state']=False
                
                bval = c2.number_input(bname, 
                                value=ses[bkey], 
                                disabled=ses[bkey + '_button_state'])
                ses[bkey] = bval
        st.write('')            
    
    st.divider()

    #tmp
    #st.write({k:ses[k] for k in ses if '_' not in k})
    

    # РАССЧИТАТЬ
    if st.button("Рассчитать"):
        biomarker_values = {k:[ses[k]] for k in ses if '_' not in k}
        biomarker_values = pd.DataFrame(biomarker_values)
        biomarker_values['gender'] = 1 if gender=='Мужской' else 0
        biomarker_values = biomarker_values[load(models_saved[0]).feature_names_]
        
        print('biomarker_values', biomarker_values)
        ba = predict_and_average(biomarker_values, models_saved)
        st.write(f"Биологический возраст: {int(ba)}")
        
    st.divider()

    st.title("FAQ")
    # Folding text with info about the principle of biological age computation
    with st.expander("Принцип расчета биологического возраста Biological Age Computation."):
        st.markdown("""
                    тут будет описание
                    """)


if __name__ == "__main__":
    main()
