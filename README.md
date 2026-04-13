# Hybrid Intrusion Detection System for Cloud Security

This project implements a hybrid intrusion detection system that combines Deep Neural Networks (DNN) and heuristic rule-based analysis to identify malicious network traffic in cloud environments.

## Overview

The system classifies network traffic into Normal or Attack using a trained DNN model, while heuristic rules are applied to detect anomalies such as traffic spikes and protocol irregularities. The goal is to improve reliability and enable real-time detection.

## Features

- End-to-end intrusion detection pipeline  
- Deep learning-based classification using DNN  
- Heuristic anomaly detection  
- Flask-based backend for real-time prediction  
- Web interface for user input  

## System Workflow

1. User inputs network features through the frontend  
2. Data is processed in the backend:
   - One-hot encoding  
   - Feature alignment  
   - Feature scaling  
3. Processed data is passed to the DNN model  
4. Heuristic rules are applied for anomaly detection  
5. Final result (Normal/Attack) is returned  

## Datasets

- NSL-KDD  
- UNSW-NB15  

## Models

- Random Forest (~80% accuracy)  
- Deep Neural Network (~79% accuracy) [Final Model]  
- LSTM (~78% accuracy)  

## Tech Stack

- Python, Flask  
- TensorFlow / Keras  
- Scikit-learn  
- HTML, CSS, JavaScript  

## Novelty

The project introduces a hybrid approach by combining deep learning with rule-based analysis, improving detection reliability over single-model systems.

## Deployment

- Backend deployed on Render  
- Frontend hosted using GitHub Pages  

## Research Work

A research paper has been drafted based on this project, including literature review, methodology, and results.

## Author

Shubh Gupta
