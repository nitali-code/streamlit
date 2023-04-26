import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Title
st.title("Webapp Using Streamlit")
 
#Image
#st.image("streamlit.png",width=500)
st.image("https://4cs.gia.edu/wp-content/uploads/2022/07/1-262298.jpg", width=500)

st.title("Case study on diamond dataset")

data = sns.load_dataset("diamonds")
st.write('Shape of a dataset',data.shape)
menu = st.sidebar.radio('Menu',['Home','Prediction Price'])
if menu == 'Home':
    #st.image('diamond.png',width=550)
    st.header('Tabular Data of a Diamond')
    if st.checkbox('Tabular Data'):
        st.table(data.head(10))
    st.header('Statistical summary of a Dataframe')
    if st.checkbox("Statistics"):
        st.table(data.describe())
    if st.header('Correlation gaph'):
        fig,ax=plt.subplots(figsize=(5,2.5))
        sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
        st.pyplot(fig)
    st.title('Graphs')
    graph=st.selectbox('Different types of graphs',["Scatter plot","Bar Graph",'Histogram'])

    if graph == "Scatter plot":
        value=st.slider("Filter data using carat",0,6)
        data=data.loc[data['carat']>=value]
        fig,ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(data=data,x='carat',y='price',hue='cut')
        st.pyplot(fig)

    if graph =="Bar Graph":
        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)

    if graph=="Histogram":
        fig,ax=plt.subplots(figsize=(5,3))
        sns.distplot(data.price,kde=True)
        st.pyplot(fig)
if menu=="Prediction Price":
    st.title("Prediction price of a Diamond")

    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    X=np.array(data["carat"]).reshape(-1,1)
    y=np.array(data['price']).reshape(-1,1)
    lr.fit(X,y)
    value=st.number_input("Carat",0.20,5.01,step=0.15) #max, min, step
    value=np.array(value).reshape(-1,1)
    prediction=lr.predict(value)[0]
    if st.button("Price Prediction($)"):
        st.write(f'{prediction}')

