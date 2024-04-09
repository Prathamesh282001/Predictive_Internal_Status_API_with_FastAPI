# Settyl Data Science And Machine Learning Engineer Task Documentation

## Introduction
This document provides an overview of the development process for the Settyl Data Science and Machine Learning Engineer Task. The task involved developing a machine learning model to predict internal status based on external status descriptions and implementing an API using FastAPI framework to expose the trained model.

## Dataset
The dataset used for this task was provided as a JSON file, containing external status descriptions and corresponding internal status labels.

## Data Preprocessing
1. **Cleaning**: 
   - The `clean_data()` function was implemented to preprocess the text data by converting it to lowercase and removing certain characters such as '/', '(', ')', ':', and "'".

## Model Development
1. **Data Loading and Preprocessing**:
   - The `dataLoading()` function loads the dataset from the provided JSON file and preprocesses the data.
   - The `dataForModel()` function prepares the data for model training by tokenizing the text data and performing one-hot encoding on the labels.

2. **Model Architecture**:
   - A feedforward neural network model was implemented using Keras with the following architecture:
     - Input layer with 100 neurons and ReLU activation function
     - Dropout layer with a dropout rate of 0.5
     - Hidden layer with 100 neurons and ReLU activation function
     - Dropout layer with a dropout rate of 0.5
     - Hidden layer with 100 neurons and ReLU activation function
     - Output layer with 15 neurons (corresponding to the number of unique internal status labels) and softmax activation function

3. **Model Training**:
   - The model was compiled with categorical cross-entropy loss function, accuracy metric, and Adam optimizer.
   - Training was performed for 100 epochs with a batch size of 32.

## API Development
1. **Endpoint**:
   - Implemented a POST endpoint `/predict` using FastAPI framework to accept external status descriptions as input and return predicted internal status labels.

2. **Request Payload**:
   - The endpoint accepts JSON payload with the following structure:
     ```json
     {
       "description": "External status description goes here"
     }
     ```

3. **Response Format**:
   - The endpoint returns a JSON response with the predicted internal status label:
     ```json
     {
       "internal_status": "Predicted internal status label"
     }
     ```

## Testing and Validation
1. **Unit Testing**:
   - Unit tests were implemented to validate individual components of the code such as data preprocessing functions, model architecture, and API endpoints.
   
2. **Integration Testing**:
   - Integration tests were conducted to ensure the functionality and accuracy of the API endpoints using sample input data.

3. **Model Evaluation**:
   - The performance of the trained model was evaluated using appropriate metrics such as accuracy, precision, and recall on a validation dataset.

## Deployment
1. **Deployment Platform**:
   - Docker is used for containerization and the application image is used for easy deployment. The API was deployed on Azure cloud platform.

2. **API URL**:
   - The deployed API endpoint URL: https://statuschecker.azurewebsites.net/docs 

## Conclusion
The development process involved data preprocessing, model development, API implementation, testing, and deployment. The trained model achieved satisfactory performance, and the deployed API is ready for use.

<img width="1440" alt="Screenshot 2024-04-09 at 8 47 46 PM" src="https://github.com/Prathamesh282001/Predictive_Internal_Status_API_with_FastAPI/assets/122107260/2377658d-62f5-4705-b825-ff06ddc9ea5d">

<img width="1440" alt="Screenshot 2024-04-09 at 8 48 24 PM" src="https://github.com/Prathamesh282001/Predictive_Internal_Status_API_with_FastAPI/assets/122107260/ab4b7ce5-f8ba-4184-9b30-42b469111c25">
