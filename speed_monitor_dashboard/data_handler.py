"""
This module handles loading data from CSV files for the dashboard.
It replaces the previous data_generator module.
"""

import pandas as pd
import os

def load_csv_data():
    """
    Load camera and speeding incident data from CSV files.
    
    Returns:
        tuple: (camera_data, speeding_data) as pandas DataFrames
    """
    # Check if data directory exists
    if not os.path.exists('data'):
        raise FileNotFoundError("Data directory not found. Please create 'data' directory with cameras.csv and incidents.csv files.")
    
    # Check if required CSV files exist
    camera_file = 'data/cameras.csv'
    incidents_file = 'data/incidents.csv'
    
    if not os.path.exists(camera_file):
        raise FileNotFoundError(f"Camera data file not found: {camera_file}")
    
    if not os.path.exists(incidents_file):
        raise FileNotFoundError(f"Incidents data file not found: {incidents_file}")
    
    # Load camera data
    camera_data = pd.read_csv(camera_file)
    
    # Load incidents data and convert timestamp to datetime
    speeding_data = pd.read_csv(incidents_file)
    speeding_data['timestamp'] = pd.to_datetime(speeding_data['timestamp'])
    speeding_data = speeding_data.sort_values('timestamp', ascending=False)
    
    return camera_data, speeding_data

def save_incident(incident_data):
    """
    Save a new incident to the incidents CSV file.
    
    Args:
        incident_data (dict): Dictionary containing the incident data
    """
    incidents_file = 'data/incidents.csv'
    
    # Convert to DataFrame
    new_incident = pd.DataFrame([incident_data])
    
    # Check if file exists
    if os.path.exists(incidents_file):
        # Append to existing file without header
        new_incident.to_csv(incidents_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        new_incident.to_csv(incidents_file, index=False)
    
    return True

def save_camera(camera_data):
    """
    Save a new camera to the cameras CSV file.
    
    Args:
        camera_data (dict): Dictionary containing the camera data
    """
    camera_file = 'data/cameras.csv'
    
    # Convert to DataFrame
    new_camera = pd.DataFrame([camera_data])
    
    # Check if file exists
    if os.path.exists(camera_file):
        # Append to existing file without header
        new_camera.to_csv(camera_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        new_camera.to_csv(camera_file, index=False)
    
    return True