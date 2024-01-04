import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.ensemble import forest
import os
import asyncio
import pandas as pd


def prediction_home(df):
    # Use 'close' column as the target variable
    target_col = 'close'

    n_prev_candles = 180#180#60
    # Function to create features and target variable for both classifier and regressor
    def create_features_and_targets(df, target_col, n):
        features = []
        target_classifier = []
        target_regressor = []

        for i in range(len(df) - n):
            features.append(df[target_col].iloc[i:i+n].values)
            target_classifier.append(np.sign(df[target_col].iloc[i+n] - df[target_col].iloc[i]))
            target_regressor.append(df[target_col].iloc[i+n])

        return np.array(features), np.array(target_classifier), np.array(target_regressor)

    # Create features and target variables for both classifier and regressor
    X, y_classifier, y_regressor = create_features_and_targets(df, target_col, n_prev_candles)

    # Split the data into training and testing sets for both tasks
    X_train, X_test, y_train_classifier, y_test_classifier, y_train_regressor, y_test_regressor = train_test_split(
        X,
        y_classifier,
        y_regressor,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Create a pipeline with StandardScaler and RandomForestRegressor
    regressor_model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

    # Use MultiOutputRegressor to combine classifier and regressor
    multioutput_regressor = MultiOutputRegressor(regressor_model)

    # Train the model
    multioutput_regressor.fit(X_train, np.column_stack((y_train_classifier, y_train_regressor)))

    # Make predictions on the test set
    predictions = multioutput_regressor.predict(X_test)

    # Extract predictions for classifier and regressor
    predictions_classifier = np.sign(predictions[:, 0])
    predictions_regressor = predictions[:, 1]

    # Evaluate the model
    accuracy_classifier = accuracy_score(y_test_classifier, predictions_classifier)
    mse_regressor = mean_squared_error(y_test_regressor, predictions_regressor)

    print(f'Classifier Accuracy: {accuracy_classifier:.24f}')
    print(f'Regressor MSE: {mse_regressor:.26f}')

    import matplotlib.pyplot as plt

    # Create an index for the x-axis based on the number of test samples
    x_index = range(len(y_test_regressor))
    # Get the indices that would sort y_test_regressor in ascending order
    sorted_indices = np.argsort(y_test_regressor)

    # Sort both y_test_regressor and predictions_regressor based on the sorted indices
    sorted_y_test_regressor = y_test_regressor[sorted_indices]
    sorted_predictions_regressor = predictions_regressor[sorted_indices]










    # Train the model
    multioutput_regressor.fit(X, np.column_stack((y_classifier, y_regressor)))


    # Tail the last n_prev_candles from the DataFrame
    last_candles = df['close'].tail(n_prev_candles).values

    # Reshape the array to match the input shape of the model
    last_candles = last_candles.reshape(1, -1)

    # Make predictions for the next n_prev_candles time points
    future_predictions = multioutput_regressor.predict(last_candles)

    # Extract predictions for classifier and regressor
    future_predictions_classifier = np.sign(future_predictions[0, 0])
    future_predictions_regressor = future_predictions[0, 1]

    # Map classifier prediction to "sell" or "buy"
    if future_predictions_classifier == -1:
        action = 'Sell'
    elif future_predictions_classifier == 1:
        action = 'Buy'
    else:
        action = 'Hold'  # You may add a 'Hold' category for cases where the classifier doesn't strongly predict either buy or sell

    # Print the interpreted action
    print(f'Classifier Prediction suggests: {action}')
    print(f'Regressor Prediction: {future_predictions_regressor}')



    n_prev_steps =(n_prev_candles)//12
    # Initialize an array to store the predicted values
    # Set the confidence threshold for the classifier prediction
    confidence_threshold = 0.7  # Adjust this threshold as needed

    # Set the uncertainty threshold for the regressor prediction
    uncertainty_threshold = 0.2  # Adjust this threshold as needed

    # Initialize an array to store the predicted values
    predicted_values = []

    # Tail the last n_prev_candles from the DataFrame
    last_candles = df['close'].tail(n_prev_candles).values

    # Reshape the array to match the input shape of the model
    last_candles = last_candles.reshape(1, -1)

    # Make predictions for the next n_prev_steps time points
    count=0
    for step in range(n_prev_steps):
        count+=1
        # Make predictions for the current step
        future_predictions = multioutput_regressor.predict(last_candles)

        # Extract predictions for classifier and regressor
        future_predictions_classifier = np.sign(future_predictions[0, 0])
        future_predictions_regressor = future_predictions[0, 1]

        # Get the confidence of the classifier prediction (assuming the classifier outputs probabilities)
        classifier_confidence = accuracy_classifier  # Replace this with your uncertainty estimation logic for the classifier


        # Calculate prediction intervals for the regressor
        
        regressor_uncertainty=0.0
        # Append the predictions, confidence, and uncertainty to the list
        predicted_values.append((future_predictions_classifier, future_predictions_regressor, classifier_confidence, regressor_uncertainty))

        # Check the confidence and uncertainty and decide whether to act on the prediction
        if classifier_confidence >= confidence_threshold and regressor_uncertainty <= uncertainty_threshold:
            # Take action based on the prediction
            if future_predictions_classifier == -1:
                action = 'Sell'
            elif future_predictions_classifier == 1:
                action = 'Buy'
            else:
                action = 'Hold'
        else:
            action = 'Uncertain'  # Do not act on the prediction if confidence or uncertainty is above the threshold

        # Print the interpreted action and predictions
        #print(f'Step {step + 1}: Classifier Prediction: {action}, Regressor Prediction: {future_predictions_regressor}, '
        #      f'Classifier Confidence: {classifier_confidence}, Regressor Uncertainty: {regressor_uncertainty}')

        # Update last_candles for the next prediction (shift the array to the left and add the new prediction to the end)
        last_candles = np.roll(last_candles, -1)
        last_candles[0, -1] = future_predictions_regressor  # Assuming the last column represents the 'close' values
        if count==n_prev_steps:

            direction=action
            price=future_predictions_regressor
            
    print('laban',direction,future_predictions_regressor)
    return direction,price,accuracy_classifier
    

