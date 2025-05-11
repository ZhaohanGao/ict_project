# Dhaka Speeding Vehicles Monitoring Dashboard

A data dashboard for monitoring speeding vehicles with image previews, vehicle details, and location data.

## Features
- Filter incidents by camera, date range, and speed
- Interactive map showing camera locations and incidents
- Data visualizations including time series, speed distribution, and camera statistics
- Add new cameras and incident records
- View detailed incident information with images

## Setup
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

## Data Storage
The application uses CSV files for data storage:
- `data/cameras.csv`: Camera location data
- `data/incidents.csv`: Speeding incident records

## Adding Data
Use the "Data Management" section to add new cameras and incidents.
