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
        self.track_path = "data/laguna_seca_mapping.csv"
        self.track = pd.read_csv(self.track_path)
        self.g = 9.81
        self.mu = 1

    def _calculate_corner(self,row):

        multiplier = self.driver.get_multiplier()[0]
        df = self.car.downforce_curve.copy()
        df['scaled_downforce'] = df['downforce']*multiplier
        def _cornering_radius(m, g, mu, v, downforce):
            return (
                ((v**2) * m) / 
                (mu * ((g * m) + downforce))
            )            
        df['req_radius'] = _cornering_radius(
            self.car.specs['mass'], self.g, self.mu, 
            df['velocity'],
            df['scaled_downforce']
        )



        cornering_speed = np.interp(
            row['radius'],
            df['req_radius'],
            df['velocity']
        )

        result = row.copy()
        result['cornering_speed'] = cornering_speed
        result['final_velocity'] = cornering_speed
        result['time'] = result['arc_length']/result['cornering_speed']
        
        return row

    def _calculate_straight(self, row):

        print(row['initial_velocity'])
        multiplier = self.driver.get_multiplier()[1]
        df = self.car.acceleration_curve.copy()
        df['scaled_time'] = df['time'] * multiplier

    #    entry_time = np.interp(
    #        initial_velocity,
    #        df['velocity'],
    #        df['scaled_time']
    #    )
        entry_curve = df[df['velocity']>=row['initial_velocity']].copy()
        start_time = entry_curve['scaled_time'].min()
        entry_curve['lag_velocity'] = entry_curve['velocity'].shift(1)
        entry_curve['lag_time'] = entry_curve['scaled_time'].shift(1)

        entry_curve['auc'] = (
            ((entry_curve['velocity'] + entry_curve['lag_velocity']) / 2) *
            (entry_curve['scaled_time'] - entry_curve['lag_time'])
        )

        entry_curve['cumauc'] = entry_curve['auc'].fillna(0).cumsum()

        straight_segment = entry_curve[entry_curve['cumauc']>=row['distance_straight']].iloc[0].copy()
        row['time'] = straight_segment['scaled_time'] - start_time
        row['final_velocity'] = straight_segment['velocity']

        return row




    def simulate_lap(self, flying_start: bool = True):

        self.track['cornering_speed'] = 0.0
        self.track['time'] = 0.0
        self.track['initial_velocity'] = 0.0
        self.track['final_velocity'] = 0.0
        track = self.track.copy()
        if not flying_start:
            track = track[track['timed']==True].copy()

        
        for i,row in track.iterrows():
            current_pos = track.index.get_loc(i)
            if current_pos>0:
                row['initial_velocity'] = track.loc[track.index[current_pos-1],'final_velocity']
            if row['type']=="corner":
                row = self._calculate_corner(row)
            else:
                row = self._calculate_straight(row)
            # if current_pos<len(track)-1:
            #     next_idx = track.index[current_pos + 1]
            #     track.loc[next_idx,'initial_velocity'] = row['final_velocity']
            track.loc[i] = row
        track['initial_velocity'] = track['initial_velocity'] * 2.237
        track['final_velocity'] = track['final_velocity'] * 2.237
        return track








    def run_simulation(self, car_name: str, skill_level, n_laps: int = 1, flying_start: bool = True, save: bool = False) -> dict:
        df = pd.read_csv("data/cars.csv")
        

        df['full_name'] = (df['year'].astype(str) + ' ' + df['make'] + ' ' + df['model'] + ' ' + df['trim'].fillna('')).str.strip()
        
        match = df[df['full_name'].str.lower() == car_name.lower()]
        
        if match.empty:
            raise ValueError(f"Car '{car_name}' not found. Available names:\n{df['full_name'].tolist()}")
            
        car_row = match.iloc[0]

        car = CarModel(car_row)
        driver = DriverProfile(skill_level)
        self.car = car
        self.driver = driver

        lap_times = []
        all_sectors = []

        for _ in range(n_laps):
            track_results = self.simulate_lap(flying_start=flying_start)
            timed = track_results[track_results['timed'] == True]
            
            lap_time = timed['time'].sum()
            lap_times.append(lap_time)
            
            sectors = timed[['num', 'time']].set_index('num')['time'].to_dict()
            all_sectors.append(sectors)

        mean_lap = np.mean(lap_times)
        std_lap = np.std(lap_times)
        p2_5 = np.percentile(lap_times, 2.5)
        p97_5 = np.percentile(lap_times, 97.5)
        interval = (p2_5, p97_5)

        results = {
            "mean_lap_time": mean_lap,
            "std_lap_time": std_lap,
            "95_percent_interval": interval,
            "lap_times": lap_times,
            "sectors": all_sectors,
            "is_flying_start": flying_start
        }

        if save:
            self.save_results(results, car_name, skill_level)

        return results

    def save_results(self, results: dict, car_name: str, skill_level):
        """Saves results to CSV automatically."""
        Path("results").mkdir(exist_ok=True)
        filename_base = f"{car_name.replace(' ', '_')}_{skill_level.value}"
        
        summary_df = pd.DataFrame([{
            "mean_lap_time": results['mean_lap_time'],
            "std_lap_time": results['std_lap_time'],
            "p2.5": results['95_percent_interval'][0],
            "p97.5": results['95_percent_interval'][1],
            "n_laps": len(results['lap_times'])
        }])
        summary_df.to_csv(f"results/{filename_base}_summary.csv", index=False)
        
        lap_times_df = pd.DataFrame(results['lap_times'], columns=['lap_time'])
        lap_times_df.to_csv(f"results/{filename_base}_laps.csv", index=False)
        
        print(f"Saved results to results/{filename_base}_summary.csv and results/{filename_base}_laps.csv")

        


       
              


        

    


       
        



