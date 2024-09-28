# -*- coding: utf-8 -*-
"""Updated Vehicular Movement Detection Model"""

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import folium
from geopy.distance import geodesic  # For calculating distances between coordinates
import tensorflow as tf

# Check for GPU
device_name = tf.test.gpu_device_name()
print(f"Using device: {device_name}")

# KML and POS parsing
import xml.etree.ElementTree as ET

def parse_kml_pos(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    coords = []
    for placemark in root.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        coord = placemark.find('.//{http://www.opengis.net/kml/2.2}coordinates').text
        lat, lon, _ = map(float, coord.split(','))
        coords.append((lat, lon))

    return coords

# POS file parser
import re

def parse_pos_file(file_path):
    gps_data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('%'):  # Skip metadata lines
                continue
            fields = re.split(r'\s+', line.strip())
            if len(fields) >= 15:
                time = fields[0] + ' ' + fields[1]  # Combine date and time
                latitude = float(fields[2])
                longitude = float(fields[3])
                height = float(fields[4])
                q_value = int(fields[5])
                num_satellites = int(fields[6])
                sdn = float(fields[7])
                sde = float(fields[8])
                sdu = float(fields[9])
                sdne = float(fields[10])
                sdeu = float(fields[11])
                sdun = float(fields[12])
                age = float(fields[13])
                ratio = float(fields[14])
                gps_data.append({
                    'time': time,
                    'latitude': latitude,
                    'longitude': longitude,
                    'height': height,
                    'q_value': q_value,
                    'num_satellites': num_satellites,
                    'sdn': sdn,
                    'sde': sde,
                    'sdu': sdu,
                    'sdne': sdne,
                    'sdeu': sdeu,
                    'sdun': sdun,
                    'age': age,
                    'ratio': ratio
                })

    return pd.DataFrame(gps_data)

# Combining GPS data from KML and POS
def combine_gps_data(kml_coords, pos_df):
    kml_df = pd.DataFrame(kml_coords, columns=['latitude', 'longitude'])
    if 'time' in pos_df.columns:
        pos_df['time'] = pd.to_datetime(pos_df['time'])
        kml_df['time'] = pos_df['time'].iloc[:len(kml_df)]
    combined_df = pd.concat([pos_df[['time', 'latitude', 'longitude']], kml_df], ignore_index=True)
    return combined_df.drop_duplicates(subset=['time', 'latitude', 'longitude']).reset_index(drop=True)

# Speed and distance calculation
def calculate_speed_and_distance(df):
    speeds = []
    distances = []
    time_diffs = []

    for i in range(1, len(df)):
        distance = geodesic((df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']),
                            (df.loc[i, 'latitude'], df.loc[i, 'longitude'])).meters

        time_diff = (pd.to_datetime(df.loc[i, 'time']) - pd.to_datetime(df.loc[i-1, 'time'])).total_seconds()
        speed = distance / time_diff if time_diff > 0 else 0
        distances.append(distance)
        speeds.append(speed)
        time_diffs.append(time_diff)

    distances.insert(0, 0)
    speeds.insert(0, 0)
    time_diffs.insert(0, 0)

    df['distance'] = distances
    df['speed'] = speeds
    df['time_diff'] = time_diffs
    return df

# Example: Label based on speed threshold (Adjust as needed for highway/service road classification)
def label_gps_data(df):
    df['label'] = np.where(df['speed'] > 15, 1, 0)  # Label: 1 for Highway, 0 for Service Road
    return df

# Data pipeline
def data_pipeline(kml_file_path, pos_file_path):
    # Parse files
    gps_coords = parse_kml_pos(kml_file_path)
    gps_data_df = parse_pos_file(pos_file_path)

    # Combine and process data
    combined_gps_data_df = combine_gps_data(gps_coords, gps_data_df)
    combined_gps_data_df = calculate_speed_and_distance(combined_gps_data_df)
    labeled_data_df = label_gps_data(combined_gps_data_df)

    return labeled_data_df

# Train/Test Split and Model Training
def train_model(data_df):
    X = data_df[['latitude', 'longitude', 'speed', 'distance', 'time_diff']]
    y = data_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return rf

# Map visualization
def visualize_map(data_df):
    vehicle_map = folium.Map(location=[data_df['latitude'].mean(), data_df['longitude'].mean()], zoom_start=14)

    for index, row in data_df.iterrows():
        color = 'green' if row['label'] == 1 else 'red'  # Green for Highway, Red for Service Road
        folium.Marker([row['latitude'], row['longitude']],
                      popup=f"Speed: {row['speed']:.2f} m/s, Highway" if row['label'] == 1 else "Service Road",
                      icon=folium.Icon(color=color)).add_to(vehicle_map)

    return vehicle_map

# Streamlit App Integration
import streamlit as st

def app():
    st.title("Vehicular Movement Detection: Highway vs Service Road")

    # File uploaders
    kml_file = st.file_uploader("Upload KML file", type="kml")
    pos_file = st.file_uploader("Upload POS file", type="pos")

    if kml_file and pos_file:
        # Process data
        combined_data = data_pipeline(kml_file, pos_file)
        
        # Display results
        st.write("Processed Data", combined_data)
        st.map(combined_data[['latitude', 'longitude']])
        
        # Train the model
        if st.button("Train Model"):
            model = train_model(combined_data)
            st.write("Model trained successfully!")

        # Visualize the route on map
        vehicle_map = visualize_map(combined_data)
        vehicle_map.save('vehicle_route.html')  # Save the map to HTML
        st.markdown("### Visualized Route")
        st.components.v1.html(open('vehicle_route.html', 'r').read(), height=500)

# Run the Streamlit app
if __name__ == "__main__":
    app()
