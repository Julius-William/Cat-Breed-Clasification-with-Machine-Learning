import streamlit as st
import eda, predict

with st.sidebar:
    st.title('Page Navigation')
    page=st.selectbox('Pilihan Halaman',
                      ('EDA','Predict'))
    
if page=='EDA':
    eda.run()
else:
    predict.run()