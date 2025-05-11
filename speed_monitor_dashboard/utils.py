"""
Utility functions for the speeding vehicles dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64
from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url):
    """Load an image from a URL and convert to base64 for embedding."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to load image from URL: {url}, Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading image from URL: {url}, Error: {str(e)}")
        return None

def filter_data(df, camera_id=None, start_date=None, end_date=None, min_speed=None, max_speed=None):
    """Filter data based on user selections."""
    filtered_df = df.copy()
    
    if camera_id:
        filtered_df = filtered_df[filtered_df['camera_id'] == camera_id]
    
    if start_date:
        filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
    
    if end_date:
        # Add one day to include the end date fully
        end_date = end_date + timedelta(days=1)
        filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
    
    if min_speed is not None:
        filtered_df = filtered_df[filtered_df['actual_speed'] >= min_speed]
    
    if max_speed is not None:
        filtered_df = filtered_df[filtered_df['actual_speed'] <= max_speed]
    
    return filtered_df

def create_map(camera_df, incident_df=None, center=None):
    """Create a folium map with camera locations and incident markers."""
    # If no center is provided, use the mean coordinates from camera_df
    if center is None:
        center = [camera_df['latitude'].mean(), camera_df['longitude'].mean()]
    
    # Create a map
    m = folium.Map(location=center, zoom_start=12)
    
    # Add camera markers
    for _, camera in camera_df.iterrows():
        folium.Marker(
            location=[camera['latitude'], camera['longitude']],
            popup=f"Camera ID: {camera['camera_id']}<br>Speed Limit: {camera['speed_limit']} km/h",
            icon=folium.Icon(icon="video-camera", prefix="fa", color="blue")
        ).add_to(m)
    
    # Add incident markers if incident data is provided
    if incident_df is not None and not incident_df.empty:
        for _, incident in incident_df.iterrows():
            # Format timestamp for display
            timestamp_str = incident['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Create popup content
            popup_content = f"""
            <b>Camera ID:</b> {incident['camera_id']}<br>
            <b>Timestamp:</b> {timestamp_str}<br>
            <b>License Plate:</b> {incident['license_plate']}<br>
            <b>Speed Limit:</b> {incident['speed_limit']} km/h<br>
            <b>Actual Speed:</b> {incident['actual_speed']} km/h<br>
            <b>Excess:</b> {incident['speed_difference']} km/h
            """
            
            # Add marker
            folium.Marker(
                location=[incident['latitude'], incident['longitude']],
                popup=popup_content,
                icon=folium.Icon(icon="car", prefix="fa", color="red")
            ).add_to(m)
    
    return m

def create_time_series_chart(df):
    """Create a time series chart of speeding incidents."""
    # Ensure the dataframe has a timestamp column
    if 'timestamp' not in df.columns or df.empty:
        return None
    
    # Group by date and count incidents
    df_by_date = df.set_index('timestamp').resample('D').size().reset_index(name='count')
    
    # Create chart
    fig = px.line(
        df_by_date, 
        x='timestamp', 
        y='count',
        title='Speeding Incidents Over Time',
        labels={'timestamp': 'Date', 'count': 'Number of Incidents'}
    )
    
    return fig

def create_speed_distribution_chart(df):
    """Create a histogram of speed violations."""
    if df.empty:
        return None
    
    fig = px.histogram(
        df,
        x='speed_difference',
        title='Distribution of Speed Violations',
        labels={'speed_difference': 'Speed Excess (km/h)'},
        nbins=20,
        color_discrete_sequence=['firebrick']
    )
    
    return fig

def create_camera_bar_chart(df):
    """Create a bar chart showing incidents by camera."""
    if df.empty:
        return None
    
    # Group by camera ID and count incidents
    camera_counts = df['camera_id'].value_counts().reset_index()
    camera_counts.columns = ['camera_id', 'count']
    
    # Sort by count in descending order
    camera_counts = camera_counts.sort_values('count', ascending=False)
    
    fig = px.bar(
        camera_counts,
        x='camera_id',
        y='count',
        title='Incidents by Camera',
        labels={'camera_id': 'Camera ID', 'count': 'Number of Incidents'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    return fig
