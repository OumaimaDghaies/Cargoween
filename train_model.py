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
    """Nettoie tous les artefacts des précédentes exécutions"""
    try:
        # Suppression du modèle principal
        if os.path.exists("best_model.pkl"):
            os.remove("best_model.pkl")
            logger.info("Ancien modèle principal supprimé")
        
        # Suppression des artefacts MLflow
        for folder in ["mlartifacts", "mlruns"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info(f"Dossier {folder} supprimé")
                
        # Nettoyage des caches Python
        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
            
    except Exception as e:
        logger.error(f"Échec du nettoyage: {str(e)}")
        raise

def train_and_save_model():
    """Exécute le pipeline d'entraînement et sauvegarde le modèle"""
    try:
        # Nettoyage complet avant démarrage
        clean_previous_files()
        
        # Chargement des variables d'environnement
        load_dotenv()
        
        # Configuration MongoDB
        MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://dghaiesoumaima0:2QM6D3ftO5H6TxH9@cluster0.g1zvwyt.mongodb.net/?retryWrites=true&w=majority")
        DB_NAME = os.getenv("DB_NAME", "reservation")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ListTransitaire")

        logger.info("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
        
        # Exécution du pipeline
        model = run_ml_pipeline(
            MONGO_URI,
            DB_NAME,
            COLLECTION_NAME
        )
        
        if model is None:
            raise ValueError("Aucun modèle retourné par le pipeline")

        # Sauvegarde du modèle
        model_path = "best_model.pkl"
        joblib.dump(model, model_path)
        
        # Vérification robuste
        if not os.path.exists(model_path):
            raise FileNotFoundError("Échec de la sauvegarde du modèle")
            
        # Log des métadonnées
        file_stats = os.stat(model_path)
        logger.info(f"""
        Modèle sauvegardé:
        - Chemin: {os.path.abspath(model_path)}
        - Taille: {file_stats.st_size / (1024 * 1024):.2f} MB
        - Dernière modification: {datetime.fromtimestamp(file_stats.st_mtime)}
        """)
        
        # Sauvegarde de backup
        os.makedirs("model_backups", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"model_backups/best_model_{timestamp}.pkl"
        joblib.dump(model, backup_path)
        logger.info(f"Backup créé: {backup_path}")
        
        return True

    except Exception as e:
        logger.error(f"ERREUR: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("=== LANCEMENT DU SCRIPT ===")
    success = train_and_save_model()
    
    # Vérification finale
    model_path = "best_model.pkl"
    if os.path.exists(model_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        logger.info(f"SUCCÈS: Modèle disponible (modifié à {mod_time})")
    else:
        logger.error("ÉCHEC: Aucun modèle généré")
    
    exit(0 if success else 1)
