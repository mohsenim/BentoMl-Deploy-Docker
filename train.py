from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def get_xgb_model_pipeline(categorical_cols, params):
    """
    Build the model
    """
    ordinal_encoder = preprocessing.OrdinalEncoder()
    preprocess = ColumnTransformer(
        [("Ordinal-Encoder", ordinal_encoder, categorical_cols)],
        remainder="passthrough",
    )
    xgb_model = xgb.XGBRegressor(**params)
    pipeline = Pipeline([("preprocess", preprocess), ("xgb_model", xgb_model)])
    return pipeline


def load_dataset(path):
    """
    Load the datase (csv file) from `path`
    """
    df = pd.read_csv(path)
    categorical_cols = ["make", "model", "fuel", "gear", "offerType"]
    numerical_cols = ["mileage_log", "hp", "age", "price_log"]

    cols = categorical_cols + numerical_cols
    data = df[cols]

    train, test = train_test_split(data, test_size=0.20, random_state=37)
    train_x = train.drop(["price_log"], axis=1)
    train_y = train[["price_log"]]

    test_x = test.drop(["price_log"], axis=1)
    test_y = test[["price_log"]]
    return (
        train_x,
        train_y,
        test_x,
        test_y,
        categorical_cols,
        numerical_cols,
    )


def train():
    """
    Train the model
    """
    dataset_path = Path("./data/autoscout24-germany-dataset-cleaned.csv")
    train_x, train_y, test_x, test_y, categorical_cols, _ = load_dataset(dataset_path)

    params = {"max_depth": 8, "subsample": 0.7}

    pipeline = get_xgb_model_pipeline(categorical_cols=categorical_cols, params=params)
    pipeline.fit(train_x, train_y)

    # Evaluation
    pred_y = pipeline.predict(test_x)
    eval_metric = mean_squared_error(test_y, pred_y)

    result = {"mse": eval_metric, "model": pipeline}
    return result


artifacts_path = Path("./artifacts")

if __name__ == "__main__":
    result = train()
    print(f"Trained! Mean squared error (MSE) of the model: {result['mse']}")

    # save the model
    model_name = "german_car_model.pkl"
    joblib.dump(result["model"], artifacts_path / model_name)
    print(f"Model {model_name} is saved in: '{artifacts_path}'.")
