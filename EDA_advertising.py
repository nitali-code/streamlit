import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.markdown('''
# **The EDA App + ML**
This is the **EDA App + ML** created in Streamlit using the **pandas-profiling** library.
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
#prediction
    st.header('**Prediction**')
    data=df.to_numpy()
    x = data[:,:-1]
    y = data[:,-1]
    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    lr.fit(x,y)
    def load_value():
        tv=st.number_input("TV",0.7,296.4,step=5.0)
        radio=st.number_input("Radio",0.0,49.6,step=5.0)
        newspaper=st.number_input("Newspaper",0.3,114.0,step=5.0)
        return tv, radio, newspaper
    value=load_value()
    value=np.array(value).reshape(-1,3)
    prediction=lr.predict(value)[0]
    if st.button("Price Prediction($)"):
        st.write(f'{prediction}')

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
#predict
        st.header('**Prediction**')
        data=df.to_numpy()
        x = data[:,:-1]
        y = data[:,-1]
        from sklearn.linear_model import LinearRegression
        lr=LinearRegression()
        lr.fit(x,y)
        def load_value():
            a=st.number_input("a",0.0024,0.95,step=0.05)
            b=st.number_input("b",0.0023,0.99,step=0.05)
            c=st.number_input("c",0.0045,0.99,step=0.05)
            d=st.number_input("d",0.04,0.98,step=0.05)
            #e=st.number_input("e",0.027,0.99,step=0.05)
            
            return a, b, c, d
        value=load_value()
        value=np.array(value).reshape(-1,4)
        prediction=lr.predict(value)[0]
        if st.button("Price Prediction($)"):
            st.write(f'{prediction}')

