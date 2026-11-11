import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



st.title('Employee Prediction App') 

st.info("this app builds a machine learning model of Employee Prediciton !")

with st.expander('Data'):
  st.write('**Row Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/17anas/data/refs/heads/main/dataset/Employee_cleaned.csv")
  df
with st.expander('X/y'):
  st.write('**X**')
  X_raw = df.drop('LeaveOrNot', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.LeaveOrNot
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='PaymentTier',y='Age',color = 'Age')

 
#Input Features
with st.sidebar:
  st.header("Input Features")
  City = st.selectbox(("City"),('Bangalore', 'Pune', 'New Delhi'))
  Gender = st.selectbox(("Gender"),("Male","Female"))
  JoiningYear = st.slider(("JoiningYear"),2017, 2013, 2014, 2016, 2015, 2012, 2018)
  PaymentTier = st.slider(("PaymentTier"),3, 1, 2)
  Age = st.slider(("Age"),34, 28, 38, 27, 24, 22, 23, 37, 32, 39, 29, 30, 36, 31, 25, 26, 40,
       			  35, 33, 41)
  ExperienceInCurrentDomain = st.slider(("ExperienceInCurrentDomain"),0, 3, 2, 5, 1, 4, 7, 6)


  #create DataFrame for the Input Features
  data = {'City':City,
          'Gender': Gender,
          'JoiningYear': JoiningYear,
          'PaymentTier': PaymentTier,
          'Age': Age,
          'ExperienceInCurrentDomain': ExperienceInCurrentDomain
           }
  input_df = pd.DataFrame(df, index = [0])
  input_penguins = pd.concat([input_df,X_raw],axis = 0)

  #Data preparation
  #encode X
  encode = ['City','Gender','Age']
  df_penguins = pd.get_dummies(input_penguins, prefix = encode)

  X = df_penguins[1:]
  input_row = df_penguins[:1]

  #encode y
  target_mapper = {'Leave' : 0,
                   'Stay' : 1}


  def target_encode(val) :
    return target_mapper[val]

  y = y_raw.apply(target_encode)
  y
  y_raw
  


with st.expander("Data preparation"):
  st.write("**Encoded X (Input penguins)**")
  input_row
  st.write("**Encoded y** ")
  y


# model training and inference
# train the ML model
clf = RandomForestClassifier()
clf.fit(X,y)

# apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ["Leave","Stay"]



# Display predicted species

st.subheader("**Predicted Species**")
st.dataframe(df_prediction_proba,
             column_config={
               'Leave': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Stay': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
              
             }, hide_index=True)
#df_prediction_proba
penguins_species = np.array(["Leave","Stay"])
st.success(str(penguins_species[prediction][0]))





  
  
