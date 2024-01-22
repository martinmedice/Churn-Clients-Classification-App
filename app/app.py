import streamlit as st
import pickle
import pandas as pd

# Página principal do aplicativo
#Configuração da aba
st.set_page_config(page_title="Churn Prediction")

st.title('Churn Classification Model')
st.image('img/customer_churn.jpeg')
st.markdown("""
    Churn prediction is a crucial aspect of any business that aims to retain its customers.
    In the context of a machine learning prediction app, churn prediction refers to the process of identifying customers who are most likely to 
    stop using a company's products or services. The app uses historical data and machine learning algorithms to analyze patterns and 
    behaviors of past customers who have churned, and then applies this knowledge to identify the customers
    who are most likely to leave in the future. By accurately predicting customer churn, a company can take proactive steps 
    to retain its valuable customers and minimize the negative impact of churn on its bottom line.

    By using this machine learning app for churn prediction, companies can save time and resources, 
    and make more informed business decisions.
    """)


# Carregar o modelo treinado
with open('models/modelo.pkl', 'rb') as arquivo_modelo:
    model = pickle.load(arquivo_modelo)

#Upload de arquivo
data = st.file_uploader('Upload your file')

#Se houverem dados faça o seguinte:
if data:
        df_input = pd.read_csv(data)  #Leia o arquivo CSV    
        df_output = df_input.assign(
            churn=model.predict(df_input),
            churn_probability=model.predict_proba(df_input)[:,1]
            ) #adicione uma coluna chamada "churn" com o model.predict do df e outra coluna chamada churn probablity que recebe o model.predict_prova[:,1]
        
        #Escreva o nome do output
        st.markdown('Churn prediction:') 
        #Escreva o df_output
        st.write(df_output)
        #Mostre um botão para download
        st.download_button(
            label='Download CSV', data=df_output.to_csv(index=False).encode('utf-8'),
            mime='text/csv', file_name='churn_prediction.csv'
            )