#from zenml.steps import BaseParameter
from pydantic import BaseModel

class ModelNameConfig(BaseModel):

    model_name: str = "LinearRegression"