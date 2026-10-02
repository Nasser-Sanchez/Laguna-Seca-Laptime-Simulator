import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
from pathlib import Path
from typing import List, Dict, Optional

class CarModel:
    def __init__(self, row:pd.Series):
        self.id = f"{row['year']}_{row['make']}_{row['model']}_{row['trim']}"
        self.specs=row
        self.t_data, self.v_data = self._extract_acceleration_points()
        self.model_type = 'log'
        self.curve_params = {}
        self.downforce_curve = self.compute_downforce_curve()
        self.acceleration_curve = self.compute_acceleration_curve()

    @staticmethod
    def _mph_to_mps(v_mph: float):
        return v_mph * 0.44704

 

    def _extract_acceleration_points(self):
        points = []

        points.append((0,0))

        if pd.notna(self.specs['accel_0_60']):
            points.append((self.specs['accel_0_60'], self._mph_to_mps(60)))
        if pd.notna(self.specs['accel_0_100']):
                    points.append((self.specs['accel_0_100'], self._mph_to_mps(100)))
        if pd.notna(self.specs['qmile_time']) and pd.notna(self.specs['qmile_speed']):
                    points.append((self.specs['qmile_time'], self._mph_to_mps(self.specs['qmile_speed'])))
        if pd.notna(self.specs['accel_0_186']):
                    points.append((self.specs['accel_0_186'], self._mph_to_mps(186)))

        if len(points) < 2:
            raise ValueError(f"Car {self.id} has insufficient acceleration points.")

        points.sort(key=lambda x: x[0])
        t_data = np.array([p[0] for p in points])
        v_data = np.array([p[1] for p in points])

        return t_data, v_data

    def _fit_log(self):
        def _log_curve(t,a,b):
            return a*np.log(b*t+1)
        popt,pcov = curve_fit(
              _log_curve, 
              self.t_data, 
              self.v_data,
              p0=[1,1]
        )
        y_pred = _log_curve(self.t_data, *popt)
        rss = np.sum((self.v_data-y_pred)**2)
        return popt, rss
        
    def _fit_exp(self):
        v_max = self._mph_to_mps(self.specs['top_speed'])
        def exp_curve(t,k):
            return v_max * (1-np.exp(-k * t))

        popt,pcov = curve_fit(
              exp_curve,
              self.t_data,
              self.v_data,
              p0=[1]
        )
        y_pred = exp_curve(self.t_data,popt[0])
        rss = np.sum((self.v_data-y_pred)**2)
        return popt[0], rss

        
    def compute_acceleration_curve(self):

        popt, rss = self._fit_log()
        a,b = popt

        # time range for v / t curve 
        max_t = 180
        t_dense = np.linspace(0, max_t, 18000)

        v_dense = a * np.log(b * t_dense + 1)

        v_max = self._mph_to_mps(self.specs['top_speed'])

        idx = np.searchsorted(v_dense, v_max)

        t_final = t_dense[:idx+1]
        v_final = v_dense[:idx+1]

        return pd.DataFrame({'time': t_final, 'velocity': v_final})

    # def _log_function(self, multiplier):
    #     popt, rss = self._fit_log()
    #     a = popt[0]
    #     b = popt[1]
    #     return (2-multiplier)*(a*np.log(b*t+1))
        
    def compute_downforce_curve(self):

        d_f = self.specs['downforce_kg'] * 9.81
        d_v = self._mph_to_mps(self.specs['downforce_speed'])

        v_grid = np.linspace(0,
                             self._mph_to_mps(self.specs['top_speed']),
                             2000
                             )
        F_d = d_f * (v_grid / d_v)**2
        return pd.DataFrame({'velocity':v_grid, 'downforce':F_d})
        

