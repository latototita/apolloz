import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load the dataset

def prediction_home(df):

    # Use 'Close' column as the target variable
    target_col = 'close'

    # Feature scaling with MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[target_col].values.reshape(-1, 1))

    # Set prediction days and create sequences
    prediction_days = 180

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    dt = x_train

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.01, random_state=42, shuffle=False)
    # Define the parameter grid
    param_grid = {
        'kernel': ['linear'],  # You can add more kernel options if needed
        'C': [0.1, 0.5,1,1.2,1.5,1.8,2],  # Adjust these values based on your problem
        'epsilon': [0.001,0.0001,0.01, 0.1, 0.5,1]  # Adjust these values based on your problem
    }

    # Create the SVR model
    svm_model = SVR()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(svm_model, param_grid, scoring='neg_mean_squared_error', cv=5)

    # Flatten the sequences for SVM
    x_train_svm = x_train.reshape(x_train.shape[0], -1)
    # Flatten the sequences for SVM
    x_train_svm_flattened = x_train.reshape(x_train.shape[0], -1)

    # Fit the grid search to the data
    grid_search.fit(x_train_svm_flattened, y_train)

    # Print the best parameters found by the grid search
    print("Best Parameters:", grid_search.best_params_)

    # Get the best SVM model
    best_svm_model = grid_search.best_estimator_

    # Make predictions on the test set
    x_test_svm_flattened = x_test.reshape(x_test.shape[0], -1)
    y_pred = best_svm_model.predict(x_test_svm_flattened)

    """
    x_train_svm = x_train.reshape(x_train.shape[0], -1)
    # Build and train the SVM model
    model = SVR(kernel='linear',C=0.1,epsilon=0.01)
    model.fit(x_train_svm, y_train)

    # Flatten the sequences for SVM
    x_test_svm = x_test.reshape(x_test.shape[0], -1)

    # Make predictions on the test set
    y_pred = model.predict(x_test_svm)
    """
    # Inverse transform the predictions and true values to the original scale
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f'Mean Squared Error: {mse:.24f}')

    # Calculate evaluation metrics
    r_squared = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv) * 100)

    # Print evaluation metrics
    print(f"R-squared: {r_squared:.24f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.24f}")
    print(f"Mean Absolute Error (MAE): {mae:.24f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.24f}%")

    # Predict Next Day
    real_data = scaled_data[-prediction_days:]  # Assuming x_train is your input data
    real_data_flattened = real_data.reshape(1, -1)  # Reshape for prediction

    # Make predictions
    prediction = best_svm_model.predict(real_data_flattened)

    # Inverse transform the prediction to the original scale
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))


    # Identify sell and buy conditions
    sell_condition = y_test_inv[:-1] > y_test_inv[1:]  # Close below the previous price
    buy_condition = y_test_inv[:-1] < y_test_inv[1:]  # Close above the previous price

    # Filter elements for buy and sell scenarios
    sell_test = y_test_inv[:-1][sell_condition]
    sell_pred = y_pred_inv[:-1][sell_condition]

    buy_test = y_test_inv[:-1][buy_condition]
    buy_pred = y_pred_inv[:-1][buy_condition]

    # Calculate maximum absolute differences for buy and sell
    min_difference_sell = np.abs(sell_test - sell_pred).min()
    min_difference_buy = np.abs(buy_test - buy_pred).min()

    return prediction[0].item(),min_difference_buy,min_difference_sell,mse
"""
file_path = 'EURUSDm_1h.csv'
df=pd.read_csv(file_path)
take_profit,min_difference_buy,min_difference_sell,mse=prediction_home(df)
print(take_profit,min_difference_buy,min_difference_sell,mse)"""
