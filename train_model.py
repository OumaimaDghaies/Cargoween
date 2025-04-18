from ml_pipeline import run_ml_pipeline
import joblib
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_and_save_model():
    """Exécute le pipeline d'entraînement et sauvegarde le modèle"""
    try:
        # Charger les variables d'environnement
        load_dotenv()
        
        # Configuration avec validation
        MONGO_CONNECTION_STRING = "mongodb+srv://dghaiesoumaima0:2QM6D3ftO5H6TxH9@cluster0.g1zvwyt.mongodb.net/?retryWrites=true&w=majority"
        DB_NAME = os.getenv("DB_NAME", "reservation")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ListTransitaire")
        
        if not MONGO_CONNECTION_STRING:
            raise ValueError("L'URI MongoDB n'est pas configurée dans les variables d'environnement")

        logger.info("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
        logger.info(f"Connexion à MongoDB - DB: {DB_NAME}, Collection: {COLLECTION_NAME}")
        
        # Exécution du pipeline
        result = run_ml_pipeline(
            MONGO_CONNECTION_STRING,
            DB_NAME,
            COLLECTION_NAME
        )
        
        if result is None:
            logger.error("Le pipeline a retourné None, vérifiez les logs précédents")
            return False
            
        # Vérification et sauvegarde du modèle
        if os.path.exists("best_model.pkl"):
            model_size = os.path.getsize("best_model.pkl") / (1024 * 1024)  # Taille en MB
            logger.info(f"Modèle sauvegardé (taille: {model_size:.2f} MB)")
            
            # Sauvegarde supplémentaire avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"model_backups/best_model_{timestamp}.pkl"
            os.makedirs("model_backups", exist_ok=True)
            joblib.dump(joblib.load("best_model.pkl"), backup_path)
            logger.info(f"Backup du modèle créé: {backup_path}")
            
            return True
            
        logger.error("Aucun modèle n'a été sauvegardé dans best_model.pkl")
        return False

    except Exception as e:
        logger.error(f"Échec de l'entraînement: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_and_save_model()
    exit(0 if success else 1)
