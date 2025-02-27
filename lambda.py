import pandas as pd
from functions import (get_s3_data, store_s3_data, send_email, get_current_day_records, 
                       next_day_forecast, calculate_X_y_row_to_predict, 
                       calculate_prediction_probability, set_target_value_previous_day,
                       process_new_row)

EMAIL = "darius.iavorschi@gmail.com" # personal email


def lambda_handler(event, context) -> dict:
    """AWS Lambda handler function to process stock data and generate trade recommendations.

    Orchestrates the following workflow:
    1. Fetches current market data and historical records
    2. Checks for duplicate entries to prevent data redundancy
    3. Processes and enriches data with technical indicators and temporal features
    4. Generates machine learning predictions for next-day trading
    5. Sends email notifications with trading recommendations
    6. Maintains data persistence in Amazon S3

    Args:
        event: AWS Lambda event object containing trigger information
        context: AWS Lambda context object containing runtime information

    Returns:
        dict: API gateway-compatible response containing:
            - statusCode: HTTP status code
            - body: Execution result or error message

    Raises:
        System errors are caught and returned as 500 status codes
    """

    try:
        # =====================================================================
        # Data Acquisition & Validation
        # =====================================================================
        # Fetch current market data and historical records
        new_row = get_current_day_records()  # From Alpha Vantage API
        historical_df = get_s3_data()  # From persistent storage

        if len(historical_df) == 0:
            raise ValueError("Empty historical dataset - initialization required")

        # Prevent duplicate entries that can occur due to:
        # - Scheduled triggers on market holidays
        # - Lambda retry mechanisms
        # - Infrastructure-level redundancies
        if new_row.name == historical_df.index[0]:
            return {
                "statusCode": 202,
                "body": "Record already exists. Duplicate insertion prevented."
            }

        # =====================================================================
        # Data Processing & Feature Engineering
        # =====================================================================
        # 1. Integrate new data with historical records
        # 2. Generate technical indicators (moving averages)
        # 3. Create temporal features (cyclic day encoding)
        # 4. Calculate ARIMA-based forecasts
        processed_df = pd.concat([process_new_row(new_row, historical_df), historical_df])
        processed_df.index.name = 'index'

        # Create label for previous day's movement
        set_target_value_previous_day(processed_df)

        # =====================================================================
        # Model Prediction & Storage
        # =====================================================================
        # Generate next-day forecast using best ARIMA parameters
        processed_df['next_day_forecast'] = list(
            next_day_forecast(processed_df['closing_prices'][::-1]))
        
        # Persist updated dataset to S3
        store_s3_data(processed_df)

        # Prepare ML features and prediction input
        X_features, y_target, prediction_input = calculate_X_y_row_to_predict(processed_df)

        # =====================================================================
        # Notification & Response
        # =====================================================================
        # Calculate prediction probabilities and send trading recommendation
        prediction_probability = calculate_prediction_probability(X_features, y_target, prediction_input)[0]
        send_email(prediction_probability, EMAIL)

        return {"statusCode": 200, "body": "Success: Prediction completed and notification sent."}

    except Exception as error:
        print(f"Critical error in lambda_handler: {str(error)}")
        return {"statusCode": 500, "body": f"Server Error: {str(error)}"}
    
    
