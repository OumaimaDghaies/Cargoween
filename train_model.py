from ml_pipeline import run_ml_pipeline
import joblib

def train_and_save_model():
    MONGO_CONNECTION_STRING = "mongodb+srv://dghaiesoumaima0:2QM6D3ftO5H6TxH9@cluster0.g1zvwyt.mongodb.net/?retryWrites=true&w=majority"
    DB_NAME = "reservation"
    COLLECTION_NAME = "ListTransitaire"
    
    print("=== ENTRAÎNEMENT DU MODÈLE ===")
    best_model_name = run_ml_pipeline(MONGO_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)
    
    if best_model_name is not None:
        print(f"\n✅ Pipeline exécuté avec succès. Meilleur modèle: {best_model_name}")
        print("Le modèle et les préprocesseurs sont sauvegardés dans best_model.pkl")

if __name__ == "__main__":
    train_and_save_model()