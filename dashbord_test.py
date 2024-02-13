import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px

# Définir la configuration de la page Streamlit en premier

st.set_page_config(layout="wide")

st.markdown(
    f"""
    <h1 style='text-align: center; color: wite;'>Attribution du crédit</h1>
    """,
    unsafe_allow_html=True
)

ride = pd.DataFrame()

# Charger le fichier csv
ride = pd.read_csv("test_data_app.csv")

@st.cache_data
def Extraction_result(ride, selected_id):

    # Chargez les données depuis le fichier CSV
    df_pred = ride.drop(columns=ride.columns[0])
    df_columns = df_pred.columns
            
    current_id = int(selected_id)

    # Vérifier si l'ID actuel est différent du dernier ID
                
    ride_subset = df_pred.iloc[[current_id]]
    ride_data = ride_subset.to_dict(orient='records')

        
    response_data = requests.post(url="http://localhost:9696/predict",json=ride_data).json()


    prediction = response_data["prediction"]
    shap_values_data = response_data["shap_values"]
    feature_globale = response_data["feature_global"]
    probability = response_data["probability"]
        
    # Ajouter le texte en fonction de la prédiction
    if prediction == 0:
        message = "<p style='color:green; font-size:40px;'>Le crédit sera accordé</p>"
        probability_text = f"Probabilité d'accord : {probability:.2%}"
    elif prediction == 1:
        message = "<p style='color:red; font-size:40px;'>Le crédit ne sera pas accordé</p>"
        probability_text = f"Probabilité de refus : {probability:.2%}"
        
    # Calculer l'explication du modèle    
    exp = shap.Explanation(np.array(shap_values_data['values']), np.array(shap_values_data['base_values']),df_columns)
    # Obtenez les noms de fonction (features) et les valeurs SHAP moyennes
    shap_values = exp.values
    features = df_columns
    
    # Calculer le data frame informations du client sélectionné
    data_information = ride_subset
    data_information['IDENTIFIANT CLIENT'] = selected_id
    # Move the selected_id column to the front
    selected_id_column = data_information.pop('IDENTIFIANT CLIENT')
    data_information.insert(0, 'IDENTIFIANT CLIENT', selected_id_column)
    
    # Calculer le data frame local feature pour l'identifiant sélectionné
    data_dict = {'IDENTIFIANT CLIENT': [selected_id]}
    # Ajoutez les autres colonnes avec des valeurs de exp.values
    for i, feature in enumerate(features):
        data_dict[feature] = shap_values[:, i]
        # Créez le DataFrame à partir du dictionnaire
        data_local = pd.DataFrame(data_dict)
        
    data_local_pos = pd.DataFrame()
    data_local_neg = pd.DataFrame()    
    for col in data_local.iloc[:, 1:].columns:
        if (data_local[col] > 0).any():
            data_local_pos[col] = data_local[col]

        if (data_local[col] < 0).any():
            data_local_neg[col] = data_local[col]
        
    return prediction, shap_values_data, feature_globale, probability, probability_text, df_columns, message, exp, data_information, data_local, data_local_pos, data_local_neg

# Sélectionner le client
selected_id = st.selectbox("Sélectionnez l'identifiant client", ride.index, key='selectbox_id')

# Stockez les résultats dans un état de session pour les conserver entre les clics sur les boutons
if 'results' not in st.session_state:
    st.session_state.results = {}
        
prediction, shap_values_data, feature_globale, probability, probability_text, df_columns, message, exp, data_information, data_local, data_local_pos, data_local_neg = Extraction_result(ride,selected_id)
                                                                                                                
message_container, plot_container = st.columns([10, 2])

# Stockez les résultats dans l'état de session
st.session_state.results['prediction'] = prediction
st.session_state.results['shap_values_data'] = shap_values_data
st.session_state.results['probability'] = probability
st.session_state.results['probability_text'] = probability_text
st.session_state.results['feature_globale'] = feature_globale
st.session_state.results['df_columns'] = df_columns
st.session_state.results['message'] = message
st.session_state.results['exp'] = exp
st.session_state.results['data_local'] = data_local
st.session_state.results['data_information'] = data_information
st.session_state.results['data_local_pos'] = data_local_pos
st.session_state.results['data_local_neg'] = data_local_neg

# Fonction pour afficher les résultats
def display_results(results):
    prediction = results.get('prediction', None)
    shap_values_data = results.get('shap_values_data', None)
    probability = results.get('probability', None)
    probability_text = results.get('probability_text', None)
    feature_globale = results.get('feature_globale', None)
    df_columns = results.get('df_columns', None)
    message = results.get('message', None)
    exp = results.get('exp', None)
    data_local = results.get('data_local', None)
    data_information = results.get('data_information', None)

    # Affichez les résultats...
    if prediction is not None and probability is not None and message is not None:
            
        # Affichez le DataFrame de local feature
        st.subheader("Informations du Client sélectionné")
        st.write(data_information)
                                
        # Afficher un graphique local features
        fig1, ax = plt.subplots(figsize=(12, 20))
        shap.plots.waterfall(exp[0], max_display=30, show=True)
        st.sidebar.pyplot(fig1)
        st.sidebar.title("Locale feature")
        # Créer un graphique feature globale
        fig, ax = plt.subplots(figsize=(12, 20))
        sns.barplot(x="importance", y="feature", data=pd.DataFrame(feature_globale))
        st.sidebar.pyplot(fig)
        st.sidebar.title("Globale feature")
        ## Créer le graphique en barres
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(0, probability, color='green' if prediction == 0 else 'red')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        # Ajouter le texte de probabilité
        ax.text(0.5, 0.7, probability_text, fontsize=14, ha='center', va='center')
        st.pyplot(fig)

    # Affichez le texte en fonction de la prédiction dans le même conteneur
    if message:
        container = st.columns([10, 2])
        container[0].markdown(message, unsafe_allow_html=True)
        
tab1, tab2, tab3 = st.tabs(["Run Result","variables positives","variables negatives"])  # Utilisez st.columns au lieu de st.tabs
with tab1:     
    if st.button("Run Result "):
        results = st.session_state.results
        message_container, plot_container = st.columns([10, 2])
        # Affichez le message, le graphique, le texte et la probabilité en fonction de la prédiction
        display_results(results)


with tab2:
    # Contenu pour le premier onglet
    results = st.session_state.results
    st.header("Variables positives")
    selected_variable_pos = st.selectbox(
        "Sélectionnez la distribution d'une variable qui contribue positivement à l'accord du crédit:",
        data_local_pos.columns.tolist(),index=None,
   placeholder="Select your feature",
        key='selectbox_pos')
    
    try:   
        client_amount = ride[selected_variable_pos][selected_id]
        display_results(results)
        st.subheader(f'Distribution de {selected_variable_pos}')
        fig, ax = plt.subplots()
        sns.histplot(ride[selected_variable_pos], bins=50, color='gray', kde=True, ax=ax)
        plt.axvline(client_amount, color='red', linestyle='dashed', linewidth=2,
                            label=f'Client sélectionné numéro_{selected_id}')
        # Ajout de la légende
        ax.legend()
        ax.set_ylabel("Nombre de clients")
        st.pyplot(fig)
        
    except KeyError:
        # Handle the KeyError here, you can print a custom message or take other actions
        print(f"The selected key '{selected_id}' is not present in the DataFrame.")
        # Assign a default value or handle the situation accordingly
        client_amount = None  # or any default value you prefer
    
with tab3:
    results = st.session_state.results
    st.header("Variables négatives")
    selected_variable_neg = st.selectbox("Sélectionnez la distribution d'une variable qui contribue négativement à l'accord du crédit:", data_local_neg.columns.tolist(),index=None,
   placeholder="Select your feature", key='selectbox_neg')
    results = st.session_state.results

    try:
        client_amount = ride[selected_variable_neg][selected_id]
        display_results(results)
        st.subheader(f'Distribution de {selected_variable_neg}')
        fig, ax = plt.subplots()
        sns.histplot(ride[selected_variable_neg], bins=50, color='gray', kde=True, ax=ax)
        plt.axvline(client_amount, color='red', linestyle='dashed', linewidth=2, label=f'Client sélectionné numéro_{selected_id}')
        # Ajout de la légende
        ax.legend()
        ax.set_ylabel("Nombre de clients")
        st.pyplot(fig)
    except KeyError:
        # Handle the KeyError here, you can print a custom message or take other actions
        print(f"The selected key '{selected_id}' is not present in the DataFrame.")
        # Assign a default value or handle the situation accordingly
        client_amount = None  # or any default value you prefer


    



   
    