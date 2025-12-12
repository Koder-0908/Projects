from pydantic import BaseModel, Field, computed_field, field_validator
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Literal, Annotated
import pickle
import pandas as pd
import numpy as np

#MLFlow
MODEL_VERSION = '1.0.0'

tier1_city = ['Jaipur', 'Chennai', 'Mumbai', 'Hyderabad', 'Delhi', 'Chandogarh', 'Kolkata', 'Banglore']
# import the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# fastapi object
app = FastAPI()

# pydantic object

class UserInput(BaseModel):

    age: Annotated[int, Field(..., gt = 0, lt = 120, description = 'Age of client')]
    weight: Annotated[float, Field(..., gt = 0, description = 'Weight of client')]
    height: Annotated[float, Field(..., gt = 0, description = 'Height of client')]
    income_lpa: Annotated[float, Field(..., gt = 0, description = 'Annual income of client')]
    smoker: Annotated[bool, Field(..., description = 'Is client a smoker?')]
    city: Annotated[str, Field(..., description = 'City of client')]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., description = 'Age of client')]
    
    @computed_field
    @property
    def bmi(self) -> float:
        return self.height/(self.weight**2)
    
    @computed_field
    @property
    def define_city_tier(city) -> int:
        if city in tier1_city:
            return 1
        else:
            return 2

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return 'Young'
        elif self.age < 45:
            return 'Adult'
        elif self.age < 60:
            return 'Middle_aged'
        return 'Senior'
    
    @field_validator('city')
    @classmethod
    def normailse(cls, v: str) -> str:
        v = v.strip().title()
        return v


@app.get('/')
def home():
    return {'message': 'Insurance Premium API'}

@app.get('/health')
def home():
    return {'Status': 'Ok',
            'Model': MODEL_VERSION,
            'Loaded': model is not None}
        
@app.post('/predict')
def predict(data: UserInput):
    data = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'city_tier': data.define_city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation,
        'smoker': data.smoker
    }
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code = 200, content = {'predicted': prediction})
