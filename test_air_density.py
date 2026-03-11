import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import csv
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

def calculate_air_density(temp_C: float, elev_m: float, 
                              relative_humidity: float, pressure_mmHg: float) -> float:
        """
        空気密度を計算（標高、気温、気圧、湿度を考慮）
        
        Parameters:
        -----------
        temp_C : float
            気温 (deg C)
        elev_m : float
            標高 (m)
        relative_humidity : float
            相対湿度 (%)
        pressure_mmHg : float
            気圧 (mm Hg)
        
        Returns:
        --------
        float
            空気密度 (kg/m^3)
        """
        # 飽和水蒸気圧を計算：Buck equation
        svp_mmHg = 4.5841 * math.exp((18.687 - temp_C/234.5) * temp_C / (257.14 + temp_C))
        
        # 空気密度を計算 (kg/m^3)
        rho_kg_m3 = 1.2929 * (273 / (temp_C + 273)) * \
                    (pressure_mmHg * math.exp(-0.0001217 * elev_m) - 
                     0.3783 * relative_humidity * svp_mmHg / 100) / 760
        
        return rho_kg_m3
    
if __name__ == "__main__":
    temp_C = 20 # 21.1
    elev_m = 4.572
    relative_humidity = 40.0
    # pressure_mmHg = 760
    pressure_inHg = 29.92  # 大気圧 (in Hg) Barometric Pressure in inches of  Hg.  Note:  this is the "corrected" value (i.e., referred to sea level)
    pressure_mmHg = pressure_inHg * 1000 / 39.37
    print(pressure_mmHg)

    rho_kg_m3 = calculate_air_density(temp_C, elev_m, relative_humidity, pressure_mmHg)
    print(rho_kg_m3)