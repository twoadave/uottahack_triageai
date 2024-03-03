## Project Overview
The proposed project aims to revolutionize patient triage in Canada by leveraging Solace's event-driven architecture (EDA) to facilitate faster, more efficient healthcare service allocation. The system, designed to minimize wait times in emergency rooms and optimize patient flow to appropriate care settings, will employ a sophisticated AI-driven approach to assess patient needs and direct them to the most suitable healthcare provider based on the severity of their condition.

## Objective
To reduce unnecessary emergency room visits, streamline patient flow in healthcare facilities, and ensure patients receive timely and appropriate care by using an automated triaging system powered by EDA.

## Target Audience
Patients seeking immediate healthcare services.
Healthcare facilities, including emergency rooms, walk-in clinics, and family doctors in Canada.

## System Components
- Patient Interface (Mobile/Web App): Allows patients to report symptoms and receive triage recommendations.
- Healthcare Provider Dashboard: Enables healthcare providers to view incoming cases, respond to patient referrals, and update availability in real-time.
- Centralized Triage and Coordination Center: A backend system that processes patient inputs, interacts with AI for risk assessment, and communicates with healthcare providers.

## Key Features
- Symptom Input and Initial Assessment: Patients input their symptoms and answer a set of preliminary questions. This data is sent through Solace Event Broker to the AI classifier.
- AI-Powered Risk Classification: Utilizing an accuracy and efficiency optimized neural network, the system classifies the patient's condition as minimal, moderate, or severe. This classification determines the recommendation: home care, doctor visit, or emergency room.
- Real-Time Updates and Notifications: Patients receive real-time updates about their case status, including wait times and appointment details, through the app.

## Quick Documentation: 
### triage_datageneration.py:
Data generation, input number of total answers, condition, and fraction of data points we wish to use for each of training and testing our neural network.
Generates and saves training and testing datasets. Includes graphing function to view histogram of datasets.

### triage_neuralnet.py:
Imports training and testing datasets from file, contains neural network class module. Creates instance of neural network, trains and tests.
Includes function to generate multiple instances of neural network and record timing with variation of number of nodes and epochs, compares with accuracy.
Creates multiple instances of neural network, chooses network with weights resulting in higest accuracy to save for future use.

### triage_interactionnet.py:
Imports trained neural network from file, takes input from user and generates risk factor assessment.



