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

        corner = row.copy()
        multiplier = self.driver.get_multiplier()[0]
        self.car.downforce_curve['scaled_downforce'] = self.car.downforce_curve['downforce']*multiplier
        def _cornering_radius(m, g, mu, v, downforce):
            return (
                ((v**2) * m) / 
                (mu * ((g * m) + downforce))
            )            
        self.car.downforce_curve['req_radius'] = _cornering_radius(
            self.car.specs['mass'], self.g, self.mu, 
            self.car.downforce_curve['velocity'],
            self.car.downforce_curve['scaled_downforce']
        )



        cornering_speed = np.interp(
            corner['radius'],
            self.car.downforce_curve['req_radius'],
            self.car.downforce_curve['velocity']
        )


        corner['cornering_speed'] = cornering_speed
        corner['final_velocity'] = cornering_speed
        corner['time'] = corner['arc_length']/corner['cornering_speed']
        
        return corner

    def _calculate_straight(self, row, exit_speed):
        speed_multiplier = self.driver.get_multiplier()[1]
        braking_multiplier = self.driver.get_multiplier()[1]
        if isinstance(exit_speed, float) and np.isnan(exit_speed):
            exit_speed = None
        # Scale curves
        self.car.acceleration_curve['scaled_time'] = self.car.acceleration_curve['time'] * speed_multiplier
        self.car.braking_curve['scaled_time'] = self.car.braking_curve['time'] * braking_multiplier
        
        if exit_speed is None:
            # No corner ahead, accelerate as much as possible
            entry_curve = self.car.acceleration_curve[
                self.car.acceleration_curve['velocity'] >= row['initial_velocity']
            ].copy()
            start_time = entry_curve['scaled_time'].min()
            entry_curve['lag_velocity'] = entry_curve['velocity'].shift(1)
            entry_curve['lag_time'] = entry_curve['scaled_time'].shift(1)
            entry_curve['auc'] = (
                ((entry_curve['velocity'] + entry_curve['lag_velocity']) / 2) *
                (entry_curve['scaled_time'] - entry_curve['lag_time'])
            )
            entry_curve['cumauc'] = entry_curve['auc'].fillna(0).cumsum()
            straight_accel_segment = entry_curve[entry_curve['cumauc'] <= row['distance_straight']]
            row['time'] = straight_accel_segment['scaled_time'].iloc[-1] - start_time
            row['final_velocity'] = straight_accel_segment['velocity'].iloc[-1]
            return row
        
        # Binary search for the maximum speed we can reach and still brake to exit_speed
        lo = row['initial_velocity']
        hi = self.car.acceleration_curve['velocity'].max()
        
        for _ in range(50):  # enough for convergence
            mid = (lo + hi) / 2
            
            # Acceleration distance: integrate velocity-time from entry to mid
            accel_seg = self.car.acceleration_curve[
                (self.car.acceleration_curve['velocity'] >= row['initial_velocity']) &
                (self.car.acceleration_curve['velocity'] <= mid)
            ].copy()
            if len(accel_seg) >= 2:
                accel_seg['lag_v'] = accel_seg['velocity'].shift(1)
                accel_seg['lag_t'] = accel_seg['scaled_time'].shift(1)
                d_accel = (((accel_seg['velocity'] + accel_seg['lag_v']) / 2) *
                           (accel_seg['scaled_time'] - accel_seg['lag_t'])).fillna(0).sum()
            else:
                d_accel = 0
            
            # Braking distance: integrate velocity-time from exit_speed to mid
            brake_seg = self.car.braking_curve[
                (self.car.braking_curve['velocity'] >= exit_speed) &
                (self.car.braking_curve['velocity'] <= mid)
            ].copy()
            if len(brake_seg) >= 2:
                brake_seg['lag_v'] = brake_seg['velocity'].shift(1)
                brake_seg['lag_t'] = brake_seg['scaled_time'].shift(1)
                d_brake = (((brake_seg['velocity'] + brake_seg['lag_v']) / 2) *
                           (brake_seg['scaled_time'] - brake_seg['lag_t'])).fillna(0).sum()
            else:
                d_brake = 0
            
            if d_accel + d_brake < row['distance_straight']:
                lo = mid  # We have room to go faster
            else:
                hi = mid  # We need to slow down
        
        v_max = max((lo + hi) / 2, row['initial_velocity'])
        
        # Acceleration phase
        accel_seg = self.car.acceleration_curve[
            (self.car.acceleration_curve['velocity'] >= row['initial_velocity']) &
            (self.car.acceleration_curve['velocity'] <= v_max)
        ].copy()
        
        if len(accel_seg) >= 2:
            start_time = accel_seg['scaled_time'].min()
            accel_seg['lag_v'] = accel_seg['velocity'].shift(1)
            accel_seg['lag_t'] = accel_seg['scaled_time'].shift(1)
            accel_seg['auc'] = (
                ((accel_seg['velocity'] + accel_seg['lag_v']) / 2) *
                (accel_seg['scaled_time'] - accel_seg['lag_t'])
            )
            t_accel = accel_seg['scaled_time'].iloc[-1] - start_time
        else:
            t_accel = 0
        
        # Braking phase
        brake_seg = self.car.braking_curve[
            (self.car.braking_curve['velocity'] >= exit_speed) &
            (self.car.braking_curve['velocity'] <= v_max)
        ].copy()
        
        if len(brake_seg) >= 2:
            t_brake = brake_seg['scaled_time'].iloc[-1] - brake_seg['scaled_time'].iloc[0]
        else:
            t_brake = 0
        
        row['time'] = t_accel + t_brake
        row['final_velocity'] = exit_speed
        row['peak_velocity'] = v_max
        # print(f"Straight {row['num']}: dist={row['distance_straight']:.1f}, "
        #       f"init={row['initial_velocity']:.1f}, exit={exit_speed:.1f}, "
        #       f"v_max={v_max:.1f}, t_accel={t_accel:.3f}, t_brake={t_brake:.3f}")
        return row




    def simulate_lap(self, flying_start: bool=True):
        # Calculate corner limits first
        for i, row in self.track.iterrows():
            if row['type'] == "corner":
                row = self._calculate_corner(row)
                self.track.loc[i, 'cornering_speed'] = row['cornering_speed']

        # Run the lap logic
        self.track['time'] = 0.0
        self.track['initial_velocity'] = 0.0
        self.track['final_velocity'] = 0.0
        
        track = self.track.copy()
        
        if not flying_start:
            track = track[track['timed']==True].copy()

        for i, row in track.iterrows():
            current_pos = track.index.get_loc(i)
            
            # Pass velocity from previous segment
            if current_pos > 0:
                prev_idx = track.index[current_pos-1]
                prev_val = track.loc[prev_idx, 'final_velocity']
                # print(f"{row['num']} reading prev {prev_idx}: {prev_val:.1f}")
                row['initial_velocity'] = prev_val
            
            if row['type'] == "corner":
                # Set final_velocity so the next segment reads it
                row['final_velocity'] = row['cornering_speed']
                # Ensure time is set for corners
                if pd.isna(row['time']) or row['time'] == 0:
                    row['time'] = row['arc_length'] / row['cornering_speed']
            
            elif row['type'] == "straight":
                if current_pos + 1 < len(track):
                    next_row = track.iloc[current_pos + 1]
                    if next_row['type'] == 'corner':
                        corner_row = next_row.copy()
                        corner_row = self._calculate_corner(corner_row)
                        exit_speed = corner_row['cornering_speed']
                    elif pd.notna(next_row['cornering_speed']):
                        exit_speed = next_row['cornering_speed']
                    else:
                        exit_speed = None
                    row = self._calculate_straight(row, exit_speed=exit_speed)
                else:
                    row = self._calculate_straight(row, exit_speed=None)
            
            # print(f"Updating {row['num']}: final_vel={row['final_velocity']:.1f}")
            track.loc[i] = row

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
        all_track_data = []  # NEW: store full per-segment data

        for _ in range(n_laps):
            track_results = self.simulate_lap(flying_start=flying_start)
            timed = track_results[track_results['timed'] == True]
            
            lap_time = timed['time'].sum()
            lap_times.append(lap_time)
            
            sectors = timed[['num', 'time']].set_index('num')['time'].to_dict()
            all_sectors.append(sectors)
            
            all_track_data.append(timed.copy())

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
            "is_flying_start": flying_start,
            "track_data": all_track_data,  # NEW
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
        
        for lap_idx, track_df in enumerate(results['track_data']):
            lap_filename = f"results/{filename_base}_lap{lap_idx+1}_track.csv"
            track_df.to_csv(lap_filename, index=False)
        
        all_sectors = results['sectors']
        if all_sectors:
            all_keys = set()
            for s in all_sectors:
                all_keys.update(s.keys())
            sector_avgs = {}
            for key in sorted(all_keys, key=lambda x: (int(x) if x.replace('_','').isdigit() else float('inf'))):
                vals = [sectors[key] for sectors in all_sectors if key in sectors]
                sector_avgs[f'sector_{key}_avg'] = np.mean(vals)
            sector_df = pd.DataFrame([sector_avgs])
            sector_df.to_csv(f"results/{filename_base}_sector_avgs.csv", index=False)
        
        print(f"Saved results to results/{filename_base}_summary.csv, results/{filename_base}_laps.csv")
        if results['track_data']:
            print(f"  + {len(results['track_data'])} track CSV(s) with full per-segment data")
        if any(results['sectors']):
            print(f"  + sector averages: results/{filename_base}_sector_avgs.csv")