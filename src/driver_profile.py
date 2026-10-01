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
            SkillLevel.NOVICE: (25, 50, 50, 55),
            SkillLevel.PRO: (100, 1, 10000, 2),
            SkillLevel.MAX: ()
        }[self.skill_level]

    def get_multiplier(self):

        if self.skill_level == SkillLevel.MAX:
            return 1,1

        cornering = beta.rvs(self.a1, self.b1)
        straight = beta.rvs(self.a2, self.a2)

        return [2-cornering, 2-straight]