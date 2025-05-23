CREATE TABLE predictors (
    price FLOAT,
    predictions FLOAT,
    symbol STRING,
    prediction_timestamp_ms BIGINT,
    model_name STRING,
    model_version STRING,
    predicted_ts_ms BIGINT,
    PRIMARY KEY (symbol, prediction_timestamp_ms, model_name, model_version, predicted_ts_ms)
);