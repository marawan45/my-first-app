import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt   
import seaborn as sns             
import plotly.express as px       
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

import io


 
if "theme" not in st.session_state:
    st.session_state.theme = "Light"


theme_choice = st.sidebar.radio(
    "Choose Theme:",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1
)


st.session_state.theme = theme_choice


light_theme = """
<style>
/* Main App */
.stApp {
    background-color: #FDFDFD;
    color: #111111;
    font-family: "Arial", sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F4F4F4;
    color: #000000;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    font-family: "Arial", sans-serif;
}
div.stButton > button:first-child:hover {
    background-color: #45a049;
}
</style>
"""

dark_theme = """
<style>
/* Main App */
.stApp {
    background-color: white;
    color: #E6EDF3;
    font-family: "Courier New", monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1C1F26;
    color: #E6EDF3;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #1F6FEB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    font-family: "Courier New", monospace;
}
div.stButton > button:first-child:hover {
    background-color: #388BFD;
}
</style>
"""


if st.session_state.theme == "Light":
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

df=pd.read_csv('netflix_titles.csv')
df_copy=df.copy()
df_copy.drop_duplicates(inplace=True)
df_copy.dropna(subset=['country'], inplace=True)

st.title('Netflix Movies and TV Shows Analysis')
st.sidebar.title('Navigation')
#add more  metrics for average duration
#i want to create metric for total movies and tv shows ,col3and need to be3side by side in columns and appear in each page
#i want to create metric for total movies and tv shows ,col3and need to be3side by side in columns and appear in each page


st.markdown("""
    <style>
    .metric-container {
        border: 2px solid #E50914;
        border-radius: 12px;
        padding: 18px 0 10px 0;
        margin: 0 8px;
        background: #0a2342;
        box-shadow: 0 2px 8px rgba(229,9,20,0.08);
        text-align: center;
    }
    .metric-title {
        font-size: 20px;
        color: #E50914;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .metric-value-movie {
        font-size: 40px;
        color: #fff;
        font-weight: bold;
    }
    .metric-value-tv {
        font-size: 40px;
        color: #fff;
        font-weight: bold;
    }
    .metric-value-duration {
        font-size: 40px;
        color: #fff;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    total_movies = df_copy[df_copy['type'] == 'Movie'].shape[0]
    st.markdown(f'''<div class="metric-container">
        <div class="metric-title">Total Movies</div>
        <div class="metric-value-movie">{total_movies}</div>
    </div>''', unsafe_allow_html=True)
with col2:
    total_tv_shows = df_copy[df_copy['type'] == 'TV Show'].shape[0]
    st.markdown(f'''<div class="metric-container">
        <div class="metric-title">Total TV Shows</div>
        <div class="metric-value-tv">{total_tv_shows}</div>
    </div>''', unsafe_allow_html=True)
with col3:
    avg_duration = df_copy['duration'].str.extract(r'(\d+)').astype(float).mean()[0]
    st.markdown(f'''<div class="metric-container">
        <div class="metric-title">Average Duration (mins)</div>
        <div class="metric-value-duration">{avg_duration:.2f}</div>
    </div>''', unsafe_allow_html=True)



st.divider()
page=st.sidebar.selectbox('Select a page', ['Home', 'Data Overview', 'Visualizations', 'Preprocessing'])
if page=='Home':
    st.header('Welcome to the Netflix Movies and TV Shows Analysis Dashboard')
    st.write('This dashboard provides insights into the Netflix dataset, including data overview, visualizations, and preprocessing steps.')
    st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=300)
    st.write('Use the sidebar to navigate through different sections of the dashboard.')
    with st.container():
        st.write("Developed by Marwan Eslam")
        st.write("Data Source: [Kaggle - Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)")
        st.write("Technologies Used: Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit")
        st.write("Feel free to explore the dataset and the various analyses presented in this dashboard!")
    with st.spinner():
        st.write("Loading...")
        import time
        time.sleep(1)
        st.success("Ready!")
    bar=st.progress(100)
    time.sleep(1)
    st.info ('Enjoy exploring the Netflix dataset!')
    st.error('Contact:')    
elif page=='Data Overview': 
    st.header('Data Overview')
    st.subheader('Dataset Preview')
    st.dataframe(df_copy.head())

    st.subheader('Statistical Summary')
    st.write(df_copy.describe(include='all'))
    st.subheader('Missing Values')
    missing_values = df_copy.isnull().sum()
    st.write(missing_values[missing_values > 0])
    st.subheader('Duplicate Rows')
    st.write(f'Total duplicate rows: {df_copy.duplicated().sum()}')     
    if st.button('Show Duplicate Rows'):
        st.write(df_copy[df_copy.duplicated()])
elif page=='Visualizations':
    st.header('Visualizations')
    st.subheader('Content Type Distribution')
    fig1 = px.pie(df_copy, names='type', title='Distribution of Content Types')
    st.plotly_chart(fig1)
    st.subheader('Content Rating Distribution')
    fig2 = px.histogram(df_copy, x='rating', title='Distribution of Content Ratings', color='type', barmode='group')
    st.plotly_chart(fig2)
    st.subheader('Content Added Over the Years')
    df_copy['date_added'] = pd.to_datetime(df_copy['date_added'], errors='coerce')
    df_copy['year_added'] = df_copy['date_added'].dt.year
    fig3 = px.histogram(df_copy, x='year_added', title='Content Added Over the Years', color='type', barmode='group')
    st.plotly_chart(fig3)
    st.subheader('Top 10 Countries by Content Production')
    top_countries = df_copy['country'].value_counts().head(10).index
    country_counts = df_copy[df_copy['country'].isin(top_countries)]['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    fig4 = px.bar(country_counts, x='country', y='count', title='Top 10 Countries by Content Production', labels={'country': 'Country', 'count': 'Number of Titles'})
    st.plotly_chart(fig4)
    st.subheader('Content Duration Distribution')
    fig5 = px.histogram(df_copy, x='duration', title='Distribution of Content Duration', color='type', barmode='group')
    st.plotly_chart(fig5)
    st.subheader('Content by Genre')
    df_copy['listed_in'] = df_copy['listed_in'].str.split(', ')
    df_exploded = df_copy.explode('listed_in')          
    top_genres = df_exploded['listed_in'].value_counts().head(10).index
    genre_counts = df_exploded[df_exploded['listed_in'].isin(top_genres)]['listed_in'].value_counts().reset_index()
    genre_counts.columns = ['listed_in', 'count']
    fig6 = px.bar(genre_counts, x='listed_in', y='count', title='Top 10 Genres by Content', labels={'listed_in': 'Genre', 'count': 'Number of Titles'})
    st.plotly_chart(fig6)

elif page=='Preprocessing': 
    st.header('Data Preprocessing Steps')
    st.subheader('Handling Missing Values')
    st.write('Missing values in the "country" column were handled by dropping rows with missing values.')
    st.write('Other columns with missing values can be handled using imputation techniques if necessary.')
    st.subheader('Encoding Categorical Variables')
    st.write('Categorical variables such as "type", "rating", and "country" can be encoded using Label Encoding or One-Hot Encoding.')
    le = LabelEncoder()
    df_copy['type_encoded'] = le.fit_transform(df_copy['type'])
    df_copy['rating_encoded'] = le.fit_transform(df_copy['rating'].astype(str))
    df_copy['country_encoded'] = le.fit_transform(df_copy['country'])
    st.write('Example of Label Encoding:')
    st.write(df_copy[['type', 'type_encoded']].head())
    st.subheader('Feature Scaling')
    st.write('Numerical features such as "duration" can be scaled using StandardScaler or MinMaxScaler.')
    scaler = StandardScaler()
    df_copy['duration_num'] = df_copy['duration'].str.extract(r'(\d+)').astype(float)
    df_copy['duration_scaled'] = scaler.fit_transform(df_copy[['duration_num']].fillna(0))
    st.write('Example of Feature Scaling:')
    st.write(df_copy[['duration', 'duration_scaled']].head())
    st.subheader('Final Preprocessed Data Preview')
    st.dataframe(df_copy.head())

    st.write('The dataset is now preprocessed and ready for further analysis or modeling.')




    

