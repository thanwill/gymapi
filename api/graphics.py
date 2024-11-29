import pandas as pd
from dash import dcc, html
import plotly.express as px


# Definir a estrutura de análises com IDs
def get_insights_types():
    return [
        {
            "id": "agua_calorias",
            "titulo": "Relação entre Consumo de Água e Calorias Queimadas",
            "descricao": "Análise que examina como o consumo de água (water_intake_(liters)) está relacionado com a quantidade de calorias queimadas (calories_burned) durante os treinos. Isso ajuda a entender se uma maior hidratação está associada a um maior gasto calórico.",
            "grafico": "water-calories-graph"
        },
        {
            "id": "consumo_agua_tipo",
            "titulo": "Consumo Médio de Água por Tipo de Treino",
            "descricao": "Avalia o consumo médio de água (water_intake_(liters)) para cada tipo de treino (workout_type). Essa análise identifica quais tipos de treinos exigem maior ingestão de líquidos.",
            "grafico": "water-by-workout-type-graph"
        },
        {
            "id": "calorias_tipo_treino",
            "titulo": "Calorias Queimadas por Tipo de Treino",
            "descricao": "Compara a média de calorias queimadas (calories_burned) entre diferentes tipos de treino (workout_type). Essa análise ajuda a identificar quais atividades são mais eficazes para queima calórica.",
            "grafico": "calories-by-workout-type-graph"
        },
        {
            "id": "correlacao_duracao_calorias",
            "titulo": "Correlação entre Duração do Treino e Calorias Queimadas",
            "descricao": "Investiga a relação entre a duração das sessões de treino (session_duration_(hours)) e a quantidade de calorias queimadas (calories_burned). Isso determina se treinos mais longos resultam em maior gasto calórico.",
            "grafico": "duration-calories-graph"
        },
        {
            "id": "var_bpm_experiencia",
            "titulo": "Variação de BPM por Nível de Experiência",
            "descricao": "Analisa como a frequência cardíaca média (avg_bpm) varia de acordo com o nível de experiência (experience_level) dos usuários. Essa análise pode indicar melhorias no condicionamento físico com o aumento da experiência.",
            "grafico": "bpm-experience-graph"
        },
    ]

def gerar_graficos(filename, insights_ids):
    
    df = pd.read_csv(filename)
    
    graficos = []
    
    insights_types = get_insights_types()
    
    for analise_id in insights_ids:
        analise_info = next((item for item in insights_types if item["id"] == analise_id), None)
        if analise_info:
            if analise_id == "agua_calorias":
                fig = px.scatter(df, x="water_intake_(liters)", y="calories_burned", title=analise_info["titulo"])
            elif analise_id == "consumo_agua_tipo":
                fig = px.bar(df, x="workout_type", y="water_intake_(liters)", title=analise_info["titulo"])
            elif analise_id == "calorias_tipo_treino":
                fig = px.bar(df, x="workout_type", y="calories_burned", title=analise_info["titulo"])
            elif analise_id == "correlacao_duracao_calorias":
                fig = px.scatter(df, x="session_duration_(hours)", y="calories_burned", title=analise_info["titulo"])
            elif analise_id == "var_bpm_experiencia":
                fig = px.box(df, x="experience_level", y="avg_bpm", title=analise_info["titulo"])
            
            graficos.append({
                "id": analise_id,
                "titulo": analise_info["titulo"],
                "descricao": analise_info["descricao"],
                "grafico": fig.to_html(full_html=False)
            })
            
    return graficos