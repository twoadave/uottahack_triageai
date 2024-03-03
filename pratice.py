import pandas as pd
import numpy as np
import time
import json
import paho.mqtt.publish as publish

# Solace MQTT broker details
SOLACE_MQTT_HOST = "mr-connection-cfxjas61fc0.messaging.solace.cloud"
SOLACE_MQTT_PORT = 1883  # Standard MQTT port (1883 for non-SSL, 8883 for SSL)
SOLACE_MQTT_USERNAME = "solace-cloud-client"
SOLACE_MQTT_PASSWORD = "hep7p8eohnn1qoqgrvecg201i8"

# Load initial hospital data
df = pd.read_csv('hospital_status_initial.csv')

# Initialize tracking variables
df['PatientAdded'] = False  # Indicates if a new patient has been added
df['PatientJourney'] = np.inf  # Initialize PatientJourney to infinity for all

# Constants for simulation
TIME_INCREMENT = 1  # Simulate time passing in minutes
TREATMENT_RATE = 5  # Patients treated per iteration
WAIT_TIME_INCREMENT = 5  # Time passing in minutes for wait time decrement

def calculate_wait_time(row):
    """Calculates wait time based on current hospital resources and patient count."""
    IMPACTS = {
        'Doctors': -10,
        'Nurses': -5,
        'Beds': -1,
        'Ventilators': -3,
        'ECGs': -2,
        'Patients': 10,
    }
    BASE_WAIT_TIME = 30  # Base wait time in minutes

    total_impact = sum(row[resource] * impact for resource, impact in IMPACTS.items())
    wait_time = max(BASE_WAIT_TIME + total_impact, 0)  # Ensure wait time is not negative
    return wait_time

def publish_wait_times(hospital_data):
    """Publishes hospital wait times to a Solace topic."""
    publish.single("hospital/waitTimes", payload=json.dumps(hospital_data),
                   hostname=SOLACE_MQTT_HOST, port=SOLACE_MQTT_PORT,
                   auth={'username': SOLACE_MQTT_USERNAME, 'password': SOLACE_MQTT_PASSWORD})

# Main simulation loop
while True:
    # Choose hospital with minimum wait time and add a new patient if not already done
    if not df['PatientAdded'].any():
        min_wait_index = df['EstimatedWaitTimeMinutes'].idxmin()
        df.loc[min_wait_index, 'Patients'] += 1
        df.loc[min_wait_index, 'PatientAdded'] = True
        df.loc[min_wait_index, 'PatientJourney'] = df.loc[min_wait_index, 'EstimatedWaitTimeMinutes']
    
    # Simulate treatment and decrement patients based on TREATMENT_RATE
    df['Patients'] = df['Patients'] - np.random.randint(0, TREATMENT_RATE + 1, df.shape[0])
    df['Patients'] = df['Patients'].clip(lower=0)  # Ensure patient count doesn't go negative
    
    # Update wait times based on current status
    df['EstimatedWaitTimeMinutes'] = df.apply(calculate_wait_time, axis=1)
    
    # Prepare and publish hospital data
    hospital_data = df[['HospitalID', 'EstimatedWaitTimeMinutes']].to_dict(orient='records')
    publish_wait_times(hospital_data)
    
    # Decrease journey time for the added patient
    patient_hospital_index = df[df['PatientAdded']].index[0]
    df.loc[patient_hospital_index, 'PatientJourney'] -= WAIT_TIME_INCREMENT
    if df.loc[patient_hospital_index, 'PatientJourney'] <= 0:
        print("Our patient has been treated and released.")
        df.loc[patient_hospital_index, 'PatientAdded'] = False  # Reset patient added flag
    
    # Print current simulation status
    print(df[['HospitalID', 'EstimatedWaitTimeMinutes']])
    
    time.sleep(TIME_INCREMENT)  # Wait before next simulation step

# Optionally, save the final state to CSV
df.to_csv('hospital_status_updated.csv', index=False)
