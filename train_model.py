from ml_pipeline import run_ml_pipeline
import joblib
import os
from dotenv import load_dotenv

def train_and_save_model():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Configuration avec valeurs par défaut
    MONGO_CONNECTION_STRING = os.getenv("MONGO_URI", "mongodb+srv://dghaiesoumaima0:2QM6D3ftO5H6TxH9@cluster0.g1zvwyt.mongodb.net/?retryWrites=true&w=majority")
    DB_NAME = os.getenv("DB_NAME", "reservation")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ListTransitaire")
    
    print("=== ENTRAÎNEMENT DU MODÈLE ===")
    print(f"Connexion à MongoDB avec DB: {DB_NAME}, Collection: {COLLECTION_NAME}")
    
    try:
        best_model_name = run_ml_pipeline(MONGO_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)
        
        if best_model_name is not None:
            print(f"\n✅ Pipeline exécuté avec succès. Meilleur modèle: {best_model_name}")
            
            if os.path.exists("best_model.pkl"):
                print("Modèle sauvegardé dans best_model.pkl")
                return True
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model()