import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_training_data():
    data = pd.read_csv('C:/Users/pedro/Desktop/proyecto/data/features_df.csv')
    
    
    return data

def get_player_data(player_id, endpoint):
    if endpoint == "common_player_info":
        data_dict = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_dict()
    elif endpoint == "player_career_stats":
        data_dict = playercareerstats.PlayerCareerStats(player_id=player_id).get_dict()
    else:
        return 'Invalid Endpoint'

    result_sets = data_dict.get('resultSets', [])
    if not result_sets:
        print(f"No se encontraron conjuntos de resultados para el jugador {player_id}")
        return None

    data = result_sets[0].get('rowSet', [])
    headers = result_sets[0].get('headers', [])

    if not data or not headers:
        print(f"Datos o encabezados vacíos para el jugador {player_id}")
        return None

    return pd.DataFrame(data, columns=headers)
def getAverageAPI(features_api, player_info_df):
    df_per_game = features_api

    avg_stats = ['FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'doubles']
    for stat in avg_stats:
        df_per_game[stat] = df_per_game[stat] / df_per_game['GP']

    percentage_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
    for stat in percentage_stats:
        df_per_game[stat] = df_per_game[stat]

    career_averages = df_per_game[avg_stats + percentage_stats].mean()

    total_GP = df_per_game['GP'].sum()
    career_averages['Total_GP'] = total_GP

    features_api = pd.DataFrame([career_averages])
    features_api['weight'] = player_info_df['WEIGHT'].values
    features_api['height_inches'] = player_info_df['HEIGHT'].apply(lambda x: int(x.split('-')[0]) * 12 + int(x.split('-')[1]))

    return features_api


def get_display_data(player_id):
    player_info_df = get_player_data(player_id, "common_player_info")
    player_career_stats_df = get_player_data(player_id, "player_career_stats")
    
    
    features_api = player_career_stats_df.drop(columns=['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GS', 'MIN'])
    features_api['doubles'] = features_api['FGM'] - features_api['FG3M']
    features_api = features_api.drop(columns=['FGM'])
    features = getAverageAPI(features_api, player_info_df).drop(columns=['FGA', 'FG3A', 'FTM', 'FTA', 'DREB', 'OREB', 'BLK', 'FG3_PCT', 'FG_PCT', 'PTS']).rename(columns={'FG3M': 'triples', 'REB': 'rebounds', 'AST':'assists', 'STL': 'steals', 'TOV': 'lost_balls', 'PF':'fouls', 'FT_PCT':'FT_percentage', 'Total_GP':'games_played'})
    features = features.reindex(columns=['weight', 'games_played', 'assists', 'doubles', 'triples', 'rebounds', 'FT_percentage', 'steals', 'lost_balls', 'fouls', 'height_inches'])
    return features

def pipeline(player_id, model, scaler, imputer):
       
    player_info_df = get_player_data(player_id, "common_player_info")
    player_career_stats_df = get_player_data(player_id, "player_career_stats")

    features_api = player_career_stats_df.drop(columns=['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GS', 'MIN'])
    features_api['doubles'] = features_api['FGM'] - features_api['FG3M']
    features_api = features_api.drop(columns=['FGM'])
    features = getAverageAPI(features_api, player_info_df).drop(columns=['FGA', 'FG3A', 'FTM', 'FTA', 'DREB', 'OREB', 'BLK', 'FG3_PCT', 'FG_PCT', 'PTS']).rename(columns={'FG3M': 'triples', 'REB': 'rebounds', 'AST':'assists', 'STL': 'steals', 'TOV': 'lost_balls', 'PF':'fouls', 'FT_PCT':'FT_percentage', 'Total_GP':'games_played'})
    features = features.reindex(columns=['weight', 'games_played', 'assists', 'doubles', 'triples', 'rebounds', 'FT_percentage', 'steals', 'lost_balls', 'fouls', 'height_inches'])
    

    
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    result = model.predict(features_scaled)
    
    return result



def show_class_probabilities_histogram(model, scaler, imputer, player_id, feature_names=None):
    player_info_df = get_player_data(player_id, "common_player_info")
    player_career_stats_df = get_player_data(player_id, "player_career_stats")

    features_api = player_career_stats_df.drop(columns=['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GS', 'MIN'])
    features_api['doubles'] = features_api['FGM'] - features_api['FG3M']
    features_api = features_api.drop(columns=['FGM'])
    features = getAverageAPI(features_api, player_info_df).drop(columns=['FGA', 'FG3A', 'FTM', 'FTA', 'DREB', 'OREB', 'BLK', 'FG3_PCT', 'FG_PCT', 'PTS']).rename(columns={'FG3M': 'triples', 'REB': 'rebounds', 'AST':'assists', 'STL': 'steals', 'TOV': 'lost_balls', 'PF':'fouls', 'FT_PCT':'FT_percentage', 'Total_GP':'games_played'})
    features = features.reindex(columns=['weight', 'games_played', 'assists', 'doubles', 'triples', 'rebounds', 'FT_percentage', 'steals', 'lost_balls', 'fouls', 'height_inches'])

    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)

    # Obtener las probabilidades de pertenecer a cada clase
    probabilities = model.predict_proba(features_scaled)
    
    # Asumiendo que las clases son 'Guard', 'Forward', 'Center', puedes ajustar según las clases reales
    class_names = ['Center', 'Forward', 'Guard']  # Modifica esto según las clases de tu modelo

    # Crear un DataFrame con las probabilidades para cada clase
    prob_df = pd.DataFrame(probabilities, columns=class_names)
    
    # Mostrar las probabilidades en la aplicación sin el índice
    st.write("### Class probabilities:")
    prob_df.index = [''] * len(prob_df)
    st.table(prob_df)

    # Mostrar un gráfico de barras con las probabilidades
    st.write("### Class probabilities histogram:")
    avg_probabilities = prob_df.mean()  # Promedio de las probabilidades para cada clase

    # Crear un gráfico de barras
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, avg_probabilities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.gca().set_facecolor('#0e1117')  # Fondo de la página
    plt.gcf().set_facecolor('#0e1117')  # Fondo del gráfico completo
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')

    plt.xticks(color='white')  # Color de las etiquetas en el eje X
    plt.yticks(color='white')  # Color de las etiquetas en el eje Y

    
    plt.xlabel("Class", color='white')
    plt.ylabel("Probability", color='white')
    st.pyplot(plt)