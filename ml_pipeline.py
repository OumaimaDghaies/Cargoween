import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.impute import SimpleImputer
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pymongo.server_api import ServerApi
import tempfile
import os
def load_and_prepare_data(connection_string, db_name, collection_name):
    client = MongoClient(
        connection_string,
        server_api=ServerApi('1'),
        tls=True,
        tlsAllowInvalidCertificates=True  # Nécessaire dans certains environnements CI
    )
    
    try:
        client.admin.command('ping')
        print("Connexion à MongoDB établie avec succès")
        
        db = client[db_name]
        collection = db[collection_name]
        
        cursor = collection.find({})
        df = pd.DataFrame(list(cursor))
        
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
            
        df = clean_data(df)
        
        cols_to_keep = ["Postal Address", "Region", "Pays", "Ville", "Code Postal", 
                       "Prix KM", "prix arrivée", "douane", "Company Name"]
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        return df
        
    except Exception as e:
        print(f"Erreur de connexion à MongoDB: {e}")
        return None
    finally:
        client.close()

def clean_data(df):
    df = df.drop_duplicates()
    
    if 'Postal Address' in df.columns:
        df['Postal Address'] = df['Postal Address'].str.strip().fillna('Adresse Inconnue')
    
    numeric_cols = ['Prix KM', 'prix arrivée', 'douane']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace('[^\d.,]', '', regex=True)
            df[col] = df[col].str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in ['Region', 'Ville', 'Pays', 'Company Name']:
        if col in df.columns:
            df[col] = df[col].fillna('Inconnu')
    
    if 'Code Postal' in df.columns:
        df['Code Postal'] = pd.to_numeric(df['Code Postal'], errors='coerce')
        df['Code_Prefix'] = df['Code Postal'].astype(str).str[:2].fillna('00')
    
    return df

def prepare_for_ml(df, target_col='Prix KM'):
    if target_col in df.columns:
        median_price = df[target_col].median()
        y = np.where(df[target_col] > median_price, 'élevé', 'faible')
    else:
        raise ValueError(f"Colonne cible '{target_col}' introuvable")
    
    categorical_cols = ['Postal Address', 'Region', 'Pays', 'Ville', 'Code_Prefix']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    numeric_cols = [col for col in df.columns 
                   if col not in categorical_cols and col != target_col 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    features_numeric = df[numeric_cols]
    features_categorical = df[categorical_cols]
    
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    X_numeric = numeric_imputer.fit_transform(features_numeric)
    X_categorical = categorical_imputer.fit_transform(features_categorical)
    
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    X = np.concatenate([X_numeric_scaled, X_categorical], axis=1)
    feature_names = numeric_cols + categorical_cols
    
    return X, y, scaler, {'numeric': numeric_imputer, 'categorical': categorical_imputer}, encoders, feature_names

def train_and_evaluate_models(X, y, feature_names=None):
    # Configurer MLflow pour utiliser uniquement le serveur distant
    mlflow.set_tracking_uri("http://34.76.105.165:5000")
    mlflow.set_experiment("Transport_Logistique_Optimization")
    
    # Désactiver complètement le stockage local
    mlflow.set_registry_uri("http://34.76.105.165:5000")
    
    # Forcer le mode distant seulement
    os.environ['MLFLOW_TRACKING_URI'] = "http://34.76.105.165:5000"
    os.environ['MLFLOW_REGISTRY_URI'] = "http://34.76.105.165:5000"
    
    # Configuration pour Google Cloud
    client = MlflowClient(tracking_uri="http://34.76.105.165:5000")


    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    best_score = -1
    best_model = None
    best_run_id = None
    results = {}

    for name, model in models.items():
        with mlflow.start_run(nested=True, run_name=name):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision_score(y_test, y_pred, pos_label='élevé'),
                    "recall": recall_score(y_test, y_pred, pos_label='élevé')
                })

                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=f"{name.lower()}_model",
                    signature=signature,
                    registered_model_name="Transport_Logistique_Model"
                )

                results[name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'run_id': mlflow.active_run().info.run_id
                }

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = name
                    best_run_id = mlflow.active_run().info.run_id

            except Exception as e:
                print(f"Erreur avec {name}: {str(e)}")
                continue

    if best_run_id:
        client = MlflowClient()
        model_version = client.search_model_versions(f"run_id='{best_run_id}'")[0].version
        client.transition_model_version_stage(
            name="Transport_Logistique_Model",
            version=model_version,
            stage="Production"
        )
        print(f"\n✅ Meilleur modèle ({best_model}) marqué comme 'Production' dans MLflow.")

    return results, best_model, X_test, y_test

def load_production_model():
    client = MlflowClient()
    model_name = "Transport_Logistique_Model"
    
    try:
        # Alternative à get_latest_versions
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        prod_models = [v for v in latest_versions if v.current_stage == "Production"]
        
        if not prod_models:
            raise ValueError("Aucun modèle en production trouvé")
            
        prod_model = prod_models[0]
        model_uri = f"models:/{model_name}/{prod_model.version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"\n✅ Modèle en production chargé : {prod_model.run_id}")
        return model
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

def run_ml_pipeline(connection_string, db_name, collection_name):
    print("="*60)
    print("DÉMARRAGE DU PIPELINE DE MACHINE LEARNING".center(60))
    print("="*60)

    try:
        print("\nÉtape 1: Chargement des données depuis MongoDB...")
        df = load_and_prepare_data(connection_string, db_name, collection_name)
        
        if df is None or df.empty:
            print("\nImpossible de charger les données depuis MongoDB ou DataFrame vide")
            return None
            
        print("\nDonnées chargées avec succès depuis MongoDB:")
        print(f"- Nombre d'enregistrements: {len(df)}")
        print(f"- Colonnes disponibles: {list(df.columns)}")
    except Exception as e:
        print(f"\nErreur lors du chargement des données: {str(e)}")
        return None

    try:
        print("\nÉtape 2: Préparation des données pour le machine learning...")
        X, y, scaler, imputer, encoders, feature_names = prepare_for_ml(df)
        print("\n✅ Données préparées pour le ML:")
        print(f"- Nombre de caractéristiques: {len(feature_names)}")
        print(f"- Distribution des classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    except Exception as e:
        print(f"\n❌ Erreur lors de la préparation des données: {str(e)}")
        return None

    print("\n🔧 Entraînement des modèles en cours...")
    results, best_model_name, X_test, y_test = train_and_evaluate_models(X, y, feature_names)
    
    if best_model_name in results:
        print(f"\n🏆 Meilleur modèle sélectionné: {best_model_name}")
        print(f"- Accuracy (test): {results[best_model_name].get('accuracy', 'N/A'):.4f}")
        
        y_pred = results[best_model_name]['model'].predict(X_test)
        print("\n📝 Rapport de classification détaillé:")
        print(classification_report(y_test, y_pred))
    else:
        print("\n❌ Aucun modèle valide produit")
        return None

    try:
        joblib.dump({
            'model': results[best_model_name]['model'],
            'scaler': scaler,
            'feature_names': feature_names,
            'encoders': encoders
        }, "best_model.pkl")
        print("\n💾 Modèle sauvegardé localement dans 'best_model.pkl'")

        production_model = load_production_model()
        
        if production_model:
            sample_data = X[:1]
            prediction = production_model.predict(sample_data)
            print(f"\n🔮 Test de prédiction du modèle en production : {prediction}")
    except Exception as e:
        print(f"\n❌ Erreur lors de la gestion des modèles: {str(e)}")

    return df
