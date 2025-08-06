from zenml.steps import BaseParameter

class ModelNameConfig(BaseParameter):

    model_name: str = "LinearRegression"