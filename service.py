from pathlib import Path
from typing import Annotated

import bentoml
import joblib
import numpy as np
import pandas as pd
from bentoml.validators import DataframeSchema

artifacts_path = Path("./artifacts")


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class CarPricePrediction:
    def __init__(self) -> None:
        self.pipeline = joblib.load(artifacts_path / "car_price_model.pkl")

    @bentoml.api
    def predict(
        self,
        input_records: Annotated[
            pd.DataFrame,
            DataframeSchema(
                orient="records",
                columns=[
                    "make",
                    "model",
                    "fuel",
                    "gear",
                    "offerType",
                    "mileage_log",
                    "hp",
                    "age",
                ],
            ),
        ],
    ) -> np.ndarray:
        result = self.pipeline.predict(input_records)
        return result
