import pandas as pd
import numpy as np
from scipy.stats import beta
import json
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
from src.car_model import CarModel
from src.driver_profile import DriverProfile

class SimulationEngine:
    def __init__(self, car:CarModel, driver:DriverProfile):
        self.car = car
        self.driver = driver
        self.track_path = "data/laguna_seca_mappings.csv"
        self.track = pd.read_csv(self.track_path)
        self.g = 9.81
        self.mu = 1

    def _cornering_calculator(self,row):
        
        multiplier = self.driver.get_multiplier()[0]
        self.car.downforce_curve['d2'] = self.car.downforce_curve['downforce']*multiplier
        
        def _cornering_radius(m, g, mu, v, downforce):
            return np.sqrt(
                ((v**2) * m) / 
                (mu * ((g * m) + downforce))
            )
        self.car.downforce_curve['req_radius'] = _cornering_radius(
            self.car.specs['mass'], self.g, self.mu, 
            self.car.downforce_curve['velocity'],
            self.car.downforce_curve['d2']
        )


            # return np.sqrt(
            #     (mu * ((g*mass)+downforce)*r) / mass
            # )


