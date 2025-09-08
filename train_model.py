"""
Script to train the blood group prediction model.
This script can be run separately to prepare the model before starting the web application.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility functions
from utils.dataset import load_dataset
from utils.model import train_svm_model, save_model

def main():
    """Main function to train the model"""
    try:
        # Set paths
        dataset_folder = 'dataset/dataset_blood_group'
        model_path = 'models/blood_group_model.pkl'
        
        # Create directories if they don't exist
        os.makedirs(dataset_folder, exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Check if dataset exists
        if not os.path.exists(dataset_folder) or not os.listdir(dataset_folder):
            logger.error("Dataset not found. Please manually download it from Kaggle and extract to %s", dataset_folder)
            return False
        else:
            logger.info("Dataset found, proceeding with training...")
        
        # Load dataset and extract features
        logger.info("Loading dataset and extracting features...")
        X, y = load_dataset(dataset_folder, return_features=True)
        
        if len(X) == 0 or len(y) == 0:
            logger.error("No valid samples found in the dataset.")
            return False
        
        logger.info(f"Dataset loaded successfully. Shape: {X.shape}, Number of classes: {len(set(y))}")
        
        # Train the model
        logger.info("Training SVM model...")
        model_artifact, metrics = train_svm_model(X, y)
        
        # Save the model
        logger.info("Saving model to disk...")
        save_model(model_artifact, model_path)
        
        # Print results
        logger.info(f"Model trained successfully with accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved at: {model_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)