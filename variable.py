from pydantic import BaseModel
# 2. Class which describes car parameters
class carParameter(BaseModel):
    sales: float 
    horsepower: float 
    width: float 
    fuel_efficiency: float
    manufacturer:str
    vehicle_type:str
    
    