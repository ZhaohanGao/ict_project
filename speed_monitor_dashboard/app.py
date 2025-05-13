"""
Speeding Vehicles Dashboard

This application provides a data dashboard for monitoring speeding vehicles.
It displays vehicle details, camera locations, and captured images of speeding vehicles.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import plotly.express as px
import base64
from PIL import Image
import io
import requests

from data_handler import load_csv_data
from utils import (
    load_image_from_url, 
    filter_data, 
    create_map, 
    create_time_series_chart,
    create_speed_distribution_chart,
    create_camera_bar_chart
)

# Set page configuration
st.set_page_config(
    page_title="Speeding Vehicles Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing selected incident
if 'selected_incident' not in st.session_state:
    st.session_state.selected_incident = None

# Load data from CSV files
@st.cache_data(ttl=60)  # Cache for 60 seconds to allow refreshing
def load_data():
    try:
        camera_data, speeding_data = load_csv_data()
        return camera_data, speeding_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrames if loading fails
        return pd.DataFrame(), pd.DataFrame()

cameras_df, incidents_df = load_data()

# Video loading
def video_upload_ui():
    st.header("Upload Video for Overspeed Detection")
    st.write("Upload a video to detect speeding vehicles. The backend will analyze it and return an audio alert.")

    uploaded_video = st.file_uploader("Choose a traffic video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Submit for Detection"):
            with st.spinner("Analyzing... please wait..."):
                try:
                    response = requests.post("http://localhost:5000/detect", files={"video": uploaded_video})
                    if response.status_code == 200:
                        audio_path = "alert.mp3"
                        with open(audio_path, "wb") as f:
                            f.write(response.content)
                        st.success("Overspeeding analyzed. Playing audio alert:")
                        st.audio(audio_path)
                    else:
                        st.error(f"Server error: {response.status_code}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

# Header
st.title("📊 Dhaka Speeding Vehicles Monitoring Dashboard")
st.write("Monitor and analyze speeding vehicle incidents captured by traffic cameras in Dhaka, Bangladesh.")

# Create sidebar for filters
st.sidebar.header("Filters")

# Camera filter
camera_options = ["All Cameras"] + sorted(incidents_df["camera_id"].unique().tolist())
selected_camera = st.sidebar.selectbox("Select Camera", camera_options)
camera_filter = None if selected_camera == "All Cameras" else selected_camera

# Date range filter
max_date = incidents_df["timestamp"].max().date()
min_date = incidents_df["timestamp"].min().date()
default_start_date = max_date - timedelta(days=30)

start_date = st.sidebar.date_input("Start Date", default_start_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=start_date, max_value=max_date)

# Speed filter
min_speed_val = int(incidents_df["actual_speed"].min())
max_speed_val = int(incidents_df["actual_speed"].max())
speed_range = st.sidebar.slider(
    "Speed Range (km/h)", 
    min_value=min_speed_val, 
    max_value=max_speed_val, 
    value=(min_speed_val, max_speed_val)
)

# Apply filters
filtered_incidents = filter_data(
    incidents_df,
    camera_id=camera_filter,
    start_date=pd.Timestamp(start_date),
    end_date=pd.Timestamp(end_date),
    min_speed=speed_range[0],
    max_speed=speed_range[1]
)

# Display summary statistics
st.sidebar.header("Summary")
st.sidebar.metric("Total Incidents", len(filtered_incidents))
if not filtered_incidents.empty:
    avg_excess = filtered_incidents["speed_difference"].mean()
    st.sidebar.metric("Average Speed Excess", f"{avg_excess:.1f} km/h")
    max_excess = filtered_incidents["speed_difference"].max()
    st.sidebar.metric("Maximum Speed Excess", f"{max_excess:.1f} km/h")

# Create 2-column layout: Left for table, Right for map and charts
col1, col2 = st.columns([3, 2])

# Left column - Data table with incidents
with col1:
    st.header("Speeding Incidents")
    
    if filtered_incidents.empty:
        st.warning("No incidents match the selected filters.")
    else:
        # Format display data
        display_df = filtered_incidents.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["coordinates"] = display_df.apply(
            lambda row: f"{row['latitude']:.4f}, {row['longitude']:.4f}", axis=1
        )
        
        # Select columns to display
        display_columns = ["timestamp", "license_plate", "camera_id", "coordinates", 
                          "speed_limit", "actual_speed", "speed_difference"]
        
        # Create an interactive table
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=400,
            column_config={
                "timestamp": "Timestamp",
                "license_plate": "License Plate",
                "camera_id": "Camera ID",
                "coordinates": "Coordinates",
                "speed_limit": "Speed Limit (km/h)",
                "actual_speed": "Actual Speed (km/h)",
                "speed_difference": "Excess (km/h)"
            }
        )
        
        # Add a selectbox to pick an incident
        if not filtered_incidents.empty:
            incident_options = [f"{row['timestamp']} - {row['license_plate']} - {row['camera_id']}" 
                              for _, row in filtered_incidents.iterrows()]
            selected_incident_str = st.selectbox("Select an incident to view details:", incident_options)
            
            if selected_incident_str:
                # Get the index of the selected incident
                selected_idx = incident_options.index(selected_incident_str)
                # Get the full row data
                st.session_state.selected_incident = filtered_incidents.iloc[selected_idx]
    
    # Display selected incident details
    if st.session_state.selected_incident is not None:
        incident = st.session_state.selected_incident
        
        st.subheader("Incident Details")
        
        # Create two columns for details and image
        detail_col, image_col = st.columns([1, 1])
        
        with detail_col:
            # Format timestamp
            timestamp_str = incident['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Display incident details
            st.markdown(f"**Timestamp:** {timestamp_str}")
            st.markdown(f"**License Plate:** {incident['license_plate']}")
            st.markdown(f"**Camera ID:** {incident['camera_id']}")
            st.markdown(f"**Location:** {incident['latitude']:.6f}, {incident['longitude']:.6f}")
            st.markdown(f"**Speed Limit:** {incident['speed_limit']} km/h")
            st.markdown(f"**Measured Speed:** {incident['actual_speed']} km/h")
            st.markdown(f"**Excess Speed:** {incident['speed_difference']} km/h")
        
        with image_col:
            # Display the camera image
            image_data = load_image_from_url(incident['image_url'])
            if image_data:
                st.image(image_data, caption="Captured Image", use_container_width=True)
            else:
                st.error("Failed to load incident image.")

# Right column - Map and visualizations
with col2:
    # Display map
    st.header("Camera Locations & Incidents")
    
    # Create map with cameras and selected incidents
    incident_map = create_map(cameras_df, filtered_incidents)
    st_folium(incident_map, width=600, height=400)
    
    # Create visualizations tab
    st.header("Visualizations")
    
    # Time series chart
    time_chart = create_time_series_chart(filtered_incidents)
    if time_chart:
        st.plotly_chart(time_chart, use_container_width=True)
    else:
        st.info("Not enough data for time series visualization.")
    
    # Speed distribution chart
    speed_chart = create_speed_distribution_chart(filtered_incidents)
    if speed_chart:
        st.plotly_chart(speed_chart, use_container_width=True)
    
    # Camera incident distribution
    camera_chart = create_camera_bar_chart(filtered_incidents)
    if camera_chart:
        st.plotly_chart(camera_chart, use_container_width=True)

# Add expander for data management
st.markdown("---")
with st.expander("数据管理 / Data Management"):
    tab1, tab2 = st.tabs(["添加摄像头 / Add Camera", "添加超速事件 / Add Incident"])
    
    with tab1:
        st.subheader("添加新的摄像头 / Add New Camera")
        
        # Form for adding a new camera
        with st.form("add_camera_form"):
            camera_id = st.text_input("摄像头ID / Camera ID", value="CAM")
            location_name = st.text_input("位置名称 / Location Name")
            lat = st.number_input("纬度 / Latitude", value=23.8103, format="%.6f")
            lon = st.number_input("经度 / Longitude", value=90.4125, format="%.6f")
            speed_limit = st.number_input("速度限制 (km/h) / Speed Limit", value=60, min_value=20, max_value=120, step=10)
            
            submit_camera = st.form_submit_button("提交 / Submit")
            
            if submit_camera:
                from data_handler import save_camera
                try:
                    camera_data = {
                        "camera_id": camera_id,
                        "latitude": lat,
                        "longitude": lon,
                        "location_name": location_name,
                        "speed_limit": speed_limit
                    }
                    save_camera(camera_data)
                    st.success("摄像头已添加！请刷新页面以查看更新。/ Camera added! Refresh page to see updates.")
                    # Clear the cache to force data reload
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"添加摄像头时出错 / Error adding camera: {str(e)}")
    
    with tab2:
        st.subheader("添加超速事件 / Add Speeding Incident")
        
        # Form for adding a new incident
        with st.form("add_incident_form"):
            # Get list of cameras for selection
            camera_ids = cameras_df["camera_id"].tolist() if not cameras_df.empty else []
            
            selected_camera = st.selectbox("选择摄像头 / Select Camera", camera_ids if camera_ids else ["No cameras available"])
            license_plate = st.text_input("车牌号 / License Plate")
            date = st.date_input("日期 / Date")
            time_str = st.text_input("时间 (HH:MM:SS) / Time", value="12:00:00")
            # Combine date and time
            try:
                timestamp = pd.Timestamp(f"{date} {time_str}")
            except:
                timestamp = pd.Timestamp.now()
            
            # Get camera details if available
            camera_details = cameras_df[cameras_df["camera_id"] == selected_camera] if not cameras_df.empty and selected_camera != "No cameras available" else None
            
            if camera_details is not None and not camera_details.empty:
                default_lat = camera_details.iloc[0]["latitude"]
                default_lon = camera_details.iloc[0]["longitude"]
                default_speed_limit = camera_details.iloc[0]["speed_limit"]
            else:
                default_lat = 23.8103
                default_lon = 90.4125
                default_speed_limit = 60
            
            lat = st.number_input("纬度 / Latitude", value=default_lat, format="%.6f")
            lon = st.number_input("经度 / Longitude", value=default_lon, format="%.6f")
            speed_limit = st.number_input("速度限制 (km/h) / Speed Limit", value=default_speed_limit, min_value=20, max_value=120, step=10)
            actual_speed = st.number_input("实际速度 (km/h) / Actual Speed", value=default_speed_limit + 20, min_value=20, max_value=200, step=5)
            
            # Calculate speed difference
            speed_difference = actual_speed - speed_limit
            
            # Provide default image URL
            default_image = "https://pixabay.com/get/g9e1042998bf28190e0d600f2c7116b0c373915fd91e53f793c6332667e97fb005295592cdcbaabf378560377aa0f56b57ec45288299df1857074ef4b2a4c1eed_1280.jpg"
            image_url = st.text_input("图像URL / Image URL", value=default_image)
            
            submit_incident = st.form_submit_button("提交 / Submit")
            
            if submit_incident:
                from data_handler import save_incident
                try:
                    incident_data = {
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_id": selected_camera,
                        "license_plate": license_plate,
                        "latitude": lat,
                        "longitude": lon,
                        "speed_limit": speed_limit,
                        "actual_speed": actual_speed,
                        "speed_difference": speed_difference,
                        "image_url": image_url
                    }
                    save_incident(incident_data)
                    st.success("事件已添加！请刷新页面以查看更新。/ Incident added! Refresh page to see updates.")
                    # Clear the cache to force data reload
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"添加事件时出错 / Error adding incident: {str(e)}")
# Footer
video_upload_ui()
st.markdown("---")
st.markdown("© 2025 Dhaka Speeding Vehicles Monitoring System")
