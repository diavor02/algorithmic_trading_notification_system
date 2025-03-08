import requests
import smtplib
import ssl
import io
import boto3
from email.message import EmailMessage
import pandas as pd
import xgboost as xgb
from botocore.exceptions import NoCredentialsError
from statistics import mean
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from typing import Optional, Tuple, Dict, List


price_moving_average_periods = [2, 4, 7, 11, 16, 22, 30]
volume_moving_average_periods = [2, 4, 7]


def get_alpha_vantage_data(symbol: str = 'SO', api_key: str = '') -> Optional[Dict]:
    """Retrieve daily stock data from Alpha Vantage API.

    Args:
        symbol: Stock ticker symbol (default: 'SO')
        api_key: Alpha Vantage API key (default: demo key)

    Returns:
        Dictionary of daily time series data or None if error occurs

    Note:
        The Southern Company (SO) is chosen for its stable growth patterns, 
        which imporves model performance. Returns the last 100 daily records 
        as specified by the 'compact' output size.
    """
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'compact'
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        print("Error:", data.get('Error Message', 'Unknown error'))
        return None

    return data['Time Series (Daily)']


def get_current_day_records() -> Optional[pd.Series]:
    """Retrieve current day's stock records using Alpha Vantage API.

    Returns:
        Pandas Series with closing price and volume, or None if error occurs

    Note:
        Uses default parameters from get_alpha_vantage_data (symbol='SO')
    """
    data = get_alpha_vantage_data()
    if data is None:
        return None
        
    df_new = pd.DataFrame(data)
    if df_new.empty:
        return None

    try:
        return df_new.iloc[:, 0][['4. close', '5. volume']]\
                   .rename({'4. close': 'closing_prices', '5. volume': 'volume'})
    except KeyError:
        print("Error parsing Alpha Vantage response")
        return None


def get_s3_data(bucket_name: str = "algtradingbucket", 
                object_key: str = "historical_data.csv") -> Optional[pd.DataFrame]:
    """Retrieve historical data from S3 bucket.

    Args:
        bucket_name: S3 bucket name (default: 'algtradingbucket')
        object_key: S3 object key/path (default: 'historical_data.csv')

    Returns:
        DataFrame containing historical records or None if error occurs
    """
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_data = response["Body"].read().decode("utf-8")
        return pd.read_csv(io.StringIO(csv_data), index_col='index')
    except Exception as e:
        print(f"Error retrieving S3 data: {e}")
        return None


def get_rolling_prices(df: pd.DataFrame, 
                      periods: List[int], 
                      current_price: float) -> List[float]:
    """Calculate rolling price averages for given periods.

    Args:
        df: the dataframe containing historical values
        periods: List of window sizes for moving averages
        current_price: Latest closing price

    Returns:
        List of moving averages for each specified period
    """
    
    return [mean([current_price] + df['closing_prices'].head(p-1).tolist())
            for p in periods]


def get_rolling_volumes(df: pd.DataFrame, 
                        periods: List[int], 
                        current_volume: float) -> List[float]:
    """Calculate rolling volume averages for given periods.

    Args:
        df: the dataframe containing historical values
        periods: List of window sizes for moving averages
        current_volume: Latest trading volume

    Returns:
        List of moving averages for each specified period
    """
    return [mean([current_volume] + df['volume'].head(p-1).tolist())
            for p in periods]


def sin_cos_transformation(newRow: pd.Series) -> Tuple[float, float]:
    """Convert date to cyclical sine/cosine features.

    Args:
        newRow: the series containing raw data about the corrent day

    Returns:
        Tuple of (sin(day_of_year), cos(day_of_year)) values
    """
    date = pd.to_datetime(newRow.name)
    day_of_year = date.timetuple().tm_yday
    angle = 2 * np.pi * day_of_year / 252  # 252 trading days/year
    
    return np.sin(angle), np.cos(angle)


def process_new_row(newRow, df):
    """Processes a new data row to generate features for model input.
    
    Transforms the raw data by:
    1) Applying log transformation to volume data
    2) Calculating moving averages for closing prices and volumes
    3) Encoding cyclical date features using sine-cosine transformation

    Args:
        newRow: the series containing raw data about the corrent day
        df: the dataframe containing historical values

    Returns:
        pd.DataFrame: A new dataframe row with transformed features ready 
            for model input.
    """

    newRow['volume'] = np.log(float(newRow['volume']))

    MA_prices = get_rolling_prices(df, 
                            price_moving_average_periods, 
                            float(newRow['closing_prices']))

    MA_volumes = get_rolling_volumes(df, 
                                volume_moving_average_periods, 
                                float(newRow['volume']))

    day_sin, day_cos = sin_cos_transformation(newRow)

    new_row = pd.DataFrame([{"closing_prices": float(newRow.closing_prices), 
                                "volume": newRow.volume, "mean_2": MA_prices[0], 
                                "mean_4": MA_prices[1], "mean_7": MA_prices[2], 
                                "mean_11": MA_prices[3], "mean_16":MA_prices[4],
                                "mean_22": MA_prices[5], "mean_30": MA_prices[6], 
                                "vol_2": MA_volumes[0], "vol_4": MA_volumes[1], 
                                "vol_7": MA_volumes[2], "day_sin": day_sin, 
                                "day_cos": day_cos, "Target": -1}], 
                                index=[newRow.name])
    
    return new_row


def set_target_value_previous_day(df):
    """Updates target value for the most recent complete record based on 
    current data.
    
    Modifies the target column (-1 default) to reflect price movement:
    - Sets to 1 if current closing price increased compared to previous day
    - Sets to 0 if current closing price decreased or stayed the same
    
    Args:
        df: the dataframe containing historical values. Modified in place.

    Returns:
        None
    """

    if float(df.iloc[1, 0]) > float(df.iloc[0, 0]):
        df.iloc[1, 14] = 0
    else:
        df.iloc[1, 14] = 1


def store_s3_data(df: pd.DataFrame, 
                  bucket_name: str = "algtradingbucket", 
                  object_key: str = "historical_data.csv") -> None:
    """Store DataFrame to S3 bucket as CSV.

    Args:
        df: the dataframe containing historical values (now updated)
            Its features are: 
            - closing_prices
            - volume 
            - mean_2, 4, 7, 11, 16, 22, 30 (representing price moving averages)
            - volume_2, 4, 7 (representing volume moving averages)
            - day_sin, day_cos: representing the sin cos transformations
            - the target variable that we want to predict 
        bucket_name: S3 bucket name (default: 'algtradingbucket')
        object_key: S3 object key/path (default: 'historical_data.csv')
    """
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=csv_buffer.getvalue()
        )
        print(f"Successfully uploaded {object_key} to {bucket_name}")
    except Exception as e:
        print(f"Error storing S3 data: {e}")


def next_day_forecast(historical_prices: pd.Series) -> np.ndarray:
    """Generate next-day price forecast using ARIMA model.

    Args:
        historical_prices: Series of historical closing prices

    Returns:
        Array of reversed ARIMA predictions for technical analysis
    """

    historical_prices = historical_prices.reset_index(drop=True)

    model = ARIMA(historical_prices.astype(float), order=(2, 1, 3))
    model_fit = model.fit()
    return np.flip(model_fit.predict(start=2, end=len(historical_prices)+1))


def calculate_X_y_row_to_predict(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features and target for model training/prediction.

    Args:
        df: the dataframe containing historical values 

    Returns:
        Tuple containing:
        - Modified feature DataFrame
        - Target Series
        - Latest row for prediction

    Note:
        Modifies the input DataFrame by adding ratio features and 
        normalizing existing features
    """
    
    df['mean_4/7'] = df['mean_4'] / df['mean_7']
    df['mean_4/11'] = df['mean_4'] / df['mean_11']
    df['mean_4/16'] = df['mean_4'] / df['mean_16']
    df['mean_11/16'] = df['mean_11'] / df['mean_16']

    y = df.iloc[1:, df.columns.get_loc('Target')]
    df.drop(columns='Target', inplace=True)

    # Normalizing the price and volume columns
    price_features = df.columns[2:9]
    df[price_features] = df[price_features].div(df['closing_prices'], axis=0)
    df['next_day_forecast'] = df['next_day_forecast'].div(df['closing_prices'], axis=0)
    
    volume_features = df.columns[9:12]
    df[volume_features] = df[volume_features].div(df['volume'], axis=0)
    
    df.drop(columns=['closing_prices'], inplace=True)

    row_to_predict = df.iloc[0:1].copy()
    df.drop(index=df.index[0], inplace=True)

    return df, y, row_to_predict


def calculate_prediction_probability(X, y, row_to_predict):
    """Use an XGBoost model for predicting the next day price movement. 

    Args:
        X: Input DataFrame
        y: target variable
        row_to_predict: the series containing the 

    Returns:
        The prediction probability for tomorrow (it can take values 
        between 0 and 1)
    """

    dtrain = xgb.DMatrix(X, label=y)

    # Using the already best-found parameters
    best_param = {
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.6,
        'colsample_bytree': 0.6, 
        'gamma': 6,
        'min_child_weight': 3,
        'objective': 'binary:logistic'
    }
    num_boost_round = 115

    final_model = xgb.train(
        best_param,
        dtrain,
        num_boost_round=num_boost_round
    )

    dpredict = xgb.DMatrix(row_to_predict)

    
    prediction_prob = final_model.predict(dpredict)

    return prediction_prob


def send_email(prediction_prob: float, email: str) -> None:
    """Send trading recommendation email based on prediction probability.

    Args:
        prediction_prob: Model's prediction probability (0-1 range)
    """
    email_sender = 'sendstockupdate@gmail.com'
    email_password = 'bdvj oexx rgeg epkf'
    email_receiver = email

    subject = 'Stock Update'
    body = f"The prediction probability was {prediction_prob:.3f}.\n"
    body += "You should BUY the stock SO for tomorrow.\nHave a nice day!" if prediction_prob > 0.5 else \
            "You should SELL the stock SO for tomorrow.\nHave a nice day!"

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())

    print("Email sent successfully")
