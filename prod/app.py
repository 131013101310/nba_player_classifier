import streamlit as st
import pandas as pd
from utils import pipeline, get_display_data, show_class_probabilities_histogram
from joblib import load

model = load('prod/model.joblib')
scaler = load('prod/scaler.joblib')  
imputer = load('prod/imputer.joblib')


def create_header():
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            align-items: center;
            background-color: #1d428a;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 2rem;
        }
        .logo-img {
            width: 200px;
            margin-right: 20px;
        }
        .title-text {
            color: white;
            font-size: 2rem;
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-container">
            <img src="https://1000marcas.net/wp-content/uploads/2019/12/NBA-Logo.png" class="logo-img">
            <h1 class="title-text">Player Position Prediction</h1>
        </div>
    """, unsafe_allow_html=True)


def display_player_stats(dataframe):
    ppg = dataframe['doubles'] + dataframe['triples']
    apg = dataframe['assists']
    rpg = dataframe['rebounds']
    height = dataframe['height_inches'].iloc[0]
    weight = dataframe['weight'].iloc[0]

    stats_df = pd.DataFrame({
        "Estadística": ["Points per game (PPG)", "Assists per game (APG)", "Rebounds per game (RPG)", "Height (inches)", 'Weight (pounds)'],
        "Valor": [f"{ppg.iloc[0]:.1f}", f"{apg.iloc[0]:.1f}", f"{rpg.iloc[0]:.1f}", f"{height}", f"{weight}"]
    })
    stats_df.index = [''] * len(stats_df)
    st.table(stats_df)


def main():
    
    create_header()

    
    common_info = pd.read_csv('data\common_player_info.csv')
    id_to_name = dict(zip(common_info['person_id'], common_info['display_first_last']))
    player_names = list(id_to_name.values())

    
    selected_name = st.selectbox("Selecciona un jugador", player_names)
    player_id = [key for key, value in id_to_name.items() if value == selected_name][0]
    
    col1, col2 = st.columns([1, 3])  

    with col1:
        
        image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        st.image(image_url, width=150)

    with col2:
        
        player_data = get_display_data(player_id)
        display_player_stats(player_data)

    if st.button('Predict'):
        result = pipeline(player_id, model, scaler, imputer)
        st.write(f"Predicted position for {selected_name} ({player_id}) was: {result[0]}")
    
    if st.button("Show probabilities histogram"):
        show_class_probabilities_histogram(model, scaler, imputer, player_id)
    

if __name__ == "__main__":
    main()