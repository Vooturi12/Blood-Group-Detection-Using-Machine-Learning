import os
from datetime import datetime

class BloodGroupPrediction:
    """
    Class to represent a blood group prediction result.
    This is just a data structure, not a database model.
    """
    def __init__(self, image_path, blood_group, confidence, timestamp=None):
        """
        Initialize a new blood group prediction.
        
        Args:
            image_path (str): Path to the fingerprint image
            blood_group (str): Predicted blood group
            confidence (float): Confidence score for the prediction
            timestamp (datetime, optional): Time of prediction
        """
        self.image_path = image_path
        self.blood_group = blood_group
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.filename = os.path.basename(image_path)
    
    def to_dict(self):
        """Convert the prediction object to a dictionary for display or serialization"""
        return {
            'image_path': self.image_path,
            'blood_group': self.blood_group,
            'confidence': self.confidence,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': self.filename
        }