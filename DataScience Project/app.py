import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time
#Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')
st.image('https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg')
st.header(' model of housing prices to predict median house values in California ')
#st.subheader('''User must enter given values to predict price:
#['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup']''')
st.sidebar.title('Select house Feature 🏠')
st.sidebar.image('https://img.pikbest.com/wp/202405/buy-a-house-and-sell-real-estate-transactions-buttons-on-housing-with-currency-background-3d-rendering_9830587.jpg!w700wp')

temp_df = pd.read_csv('california.csv')
random.seed(20)
all_values=[]

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])
import pickle

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]
import time 
value=0
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
progress_bar=st.progress(value)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place=st.empty()
place.image('https://media1.tenor.com/m/ACBGhLB0v0gAAAAC/looney-tunes-telescope.gif',width=220)
    
if price>0:
   
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body = f'Predicted Median House Price:${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)

else: 
    body = ' invalid house features values'
    st.warning(body)
    

