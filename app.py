# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
import xml.etree.ElementTree as ET
from streamlit_folium import folium_static
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Function to parse KML and POS files to extract GPS data
def parse_kml_pos(file):
    tree = ET.parse(io.BytesIO(file.read()))  # Use BytesIO to read the file
    root = tree.getroot()

    coords = []
    for placemark in root.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        coord = placemark.find('.//{http://www.opengis.net/kml/2.2}coordinates').text
        lat, lon, _ = map(float, coord.split(','))
        coords.append((lat, lon))
        
    return coords

# Function to parse POS file and extract GPS data
def parse_pos_file(file):
    gps_data = []
    for line in file.decode('utf-8').splitlines():  # Decode and split lines
        if line.startswith('%'):
            continue
        fields = line.split()
        if len(fields) >= 15:
            gps_data.append({
                'time': fields[0] + ' ' + fields[1],
                'latitude': float(fields[2]),
                'longitude': float(fields[3]),
            })
    return pd.DataFrame(gps_data)

# Function to combine GPS data from KML and POS files
def combine_gps_data(kml_coords, pos_df):
    kml_df = pd.DataFrame(kml_coords, columns=['latitude', 'longitude'])
    combined_df = pd.concat([pos_df[['time', 'latitude', 'longitude']], kml_df], ignore_index=True)
    return combined_df.drop_duplicates(subset=['time', 'latitude', 'longitude']).reset_index(drop=True)

# Function to calculate speed and distance between GPS points
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

# Function to plot speed vs time with road type as hue
def plot_speed_time_road(df):
    df['time'] = pd.to_datetime(df['time'])  # Convert time to datetime if not already
    
    sns.set(style="darkgrid")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='time', y='speed', hue='label', data=df, marker='o')
    
    plt.title('Speed vs Time with Road Type')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    plt.xticks(rotation=45)
    
    st.pyplot(plt)

# Streamlit UI
st.title("Vehicular Movement Detection")

# Upload KML and POS files
kml_file = st.file_uploader("Upload KML file", type="kml")
pos_file = st.file_uploader("Upload POS file", type="pos")

if kml_file and pos_file:
    # Read and parse KML and POS files
    kml_coords = parse_kml_pos(kml_file)
    pos_data = parse_pos_file(pos_file.read())  # Read the POS file as bytes and decode

    # Combine and process the data
    combined_data = combine_gps_data(kml_coords, pos_data)
    combined_data = calculate_speed_and_distance(combined_data)

    # Label data: "Service Road" for speed <= 15, "Highway" for speed > 15
    combined_data['label'] = np.where(combined_data['speed'] > 15, 'Highway', 'Service Road')

    # Display processed data
    st.write("Processed Data", combined_data)

    # Create a Folium map centered around the mean latitude and longitude
    vehicle_map = folium.Map(location=[combined_data['latitude'].mean(), combined_data['longitude'].mean()], zoom_start=14)

    # Add markers for each GPS point
    for index, row in combined_data.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=f"Speed: {row['speed']:.2f} m/s, Road: {row['label']}").add_to(vehicle_map)

    # Display the map in Streamlit using folium_static
    folium_static(vehicle_map)

    # Plot speed vs time with road type
    st.subheader('Speed vs Time with Road Type')
    plot_speed_time_road(combined_data)
