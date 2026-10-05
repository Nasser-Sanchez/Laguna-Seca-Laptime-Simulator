import pandas as pd
import numpy as np
from scipy.stats import beta
import json
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

class SkillLevel(Enum):
    NOVICE = "novice"
    #AMATEUR = "amateur"
    #INTERMEDIATE = "intermediate"
    PRO = "pro"
    MAX = "max"

class DriverProfile:
    def __init__(self,skill_level:SkillLevel):
        self.skill_level = skill_level
        self.a1, self.b1, self.a2, self.b2 = self._get_params()

    def _get_params(self):

        return{
            SkillLevel.NOVICE: (2, 4, 3, 3),
            SkillLevel.PRO: (2, 0.5, 10000, 2),
            SkillLevel.MAX: (1,1,1,1)
        }[self.skill_level]

    def get_multiplier(self):

        if self.skill_level == SkillLevel.MAX:
            return 1,1

        cornering = beta.rvs(self.a1, self.b1)
        straight = beta.rvs(self.a2, self.a2)

        #print(f"Multipliers: {cornering} , {2-straight}")
        return [cornering, 2-straight]