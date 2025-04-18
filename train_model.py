from ml_pipeline import run_ml_pipeline
import joblib
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import shutil

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


def clean_previous_files():
    """Supprime les fichiers précédents pour forcer une nouvelle génération"""
    if os.path.exists("best_model.pkl"):
        os.remove("best_model.pkl")
    if os.path.exists("mlartifacts"):
        shutil.rmtree("mlartifacts")
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
def train_and_save_model():
    """Exécute le pipeline d'entraînement et sauvegarde le modèle"""
    try:
        clean_previous_files()
        
        # Charger les variables d'environnement
        load_dotenv()
        
        MONGO_CONNECTION_STRING = "mongodb+srv://dghaiesoumaima0:2QM6D3ftO5H6TxH9@cluster0.g1zvwyt.mongodb.net/?retryWrites=true&w=majority"
        DB_NAME = os.getenv("DB_NAME", "reservation")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ListTransitaire")
        
        if not MONGO_CONNECTION_STRING:
            raise ValueError("L'URI MongoDB n'est pas configurée")

        logger.info("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
        logger.info(f"Connexion à MongoDB - DB: {DB_NAME}, Collection: {COLLECTION_NAME}")
        
        # Exécution du pipeline
        model = run_ml_pipeline(
            MONGO_CONNECTION_STRING,
            DB_NAME,
            COLLECTION_NAME
        )
        
        if model is None:
            raise ValueError("Le pipeline a retourné None")
            
        # Sauvegarde du nouveau modèle
        joblib.dump(model, "best_model.pkl")
        logger.info("Nouveau modèle sauvegardé dans best_model.pkl")
        
        # Vérification de la sauvegarde
        if not os.path.exists("best_model.pkl"):
            raise FileNotFoundError("Le modèle n'a pas été sauvegardé correctement")
            
        # Informations de debug
        file_size = os.path.getsize("best_model.pkl") / (1024 * 1024)  # Taille en MB
        mod_time = datetime.fromtimestamp(os.path.getmtime("best_model.pkl"))
        logger.info(f"Taille du modèle: {file_size:.2f} MB")
        logger.info(f"Dernière modification: {mod_time}")
        
        # Sauvegarde supplémentaire avec timestamp
        os.makedirs("model_backups", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"model_backups/best_model_{timestamp}.pkl"
        joblib.dump(model, backup_path)
        logger.info(f"Backup du modèle créé: {backup_path}")
        
        return True

    except Exception as e:
        logger.error(f"Échec critique de l'entraînement: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("=== LANCEMENT DU SCRIPT ===")
    success = train_and_save_model()
    
    # Debug supplémentaire
    if os.path.exists("best_model.pkl"):
        logger.info(f"Vérification finale - Modèle existe, modifié le: {datetime.fromtimestamp(os.path.getmtime('best_model.pkl'))}")
    else:
        logger.error("Vérification finale - Modèle non trouvé!")
    
    exit(0 if success else 1)
