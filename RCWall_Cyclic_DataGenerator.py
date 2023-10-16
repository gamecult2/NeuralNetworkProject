import numpy as np
import os
import csv
from Units import *
import RCWall_Cyclic_Model as rcmodel
import RCWall_Cyclic_Parameters as inputsVar
import random as rnd


# Create a folder to put the output
fileName = "full_database.csv"
ResultsDir = 'CyclicAnalysisResults'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)

hw_range = np.arange(hw_min, hw_max, tw, hw, lw,)
lw_range = np.arange(lw_min, lw_max, lw_step)
tw_range = np.arange(tw_min, tw_max, tw_step)
lbe_range = np.arange(lbe_min, lbe_max, lbe_step)
fc_range = np.arange(fc_min, fc_max, fc_step)
fy_range = np.arange(fy_min, fy_max, fy_step)
bereinfNum_range = np.arange(bereinfNum_min, bereinfNum_max, bereinfNum_step)
bereinfDiam_range = np.arange(bereinfDiam_min, bereinfDiam_max, bereinfDiam_step)
webreinfNum_range = np.arange(webreinfNum_min, webreinfNum_max, webreinfNum_step)
webreinfDiam_range = np.arange(webreinfDiam_min, webreinfDiam_max, webreinfDiam_step)
loadcoef_range = np.arange(loadcoef_min, loadcoef_max, loadcoef_step)
DisplacementStep_range = np.arange(DisplacementStep_min, DisplacementStep_max, DisplacementStep_step)

# Loop through all combinations of values
for hw in hw_range:
    for lw in lw_range:
        for tw in tw_range:
            for lbe in lbe_range:
                for fc in fc_range:
                    for fy in fy_range:
                        for bereinfNum in bereinfNum_range:
                            for bereinfDiam in bereinfDiam_range:
                                for webreinfNum in webreinfNum_range:
                                    for webreinfDiam in webreinfDiam_range:
                                        for loadcoef in loadcoef_range:
                                            for DisplacementStep in DisplacementStep_range:
                                                # Build the model with the current set of parameters
                                                rcmodel.build_model(hw, lw, tw, lbe, fc, fy, bereinfNum, bereinfDiam, webreinfNum, webreinfDiam, loadcoef)
                                                # Run gravity analysis + cyclic analysis with the specified DisplacementStep
                                                rcmodel.run_analysis(DisplacementStep, plotPushOverResults=True)
                                                # Reset the analysis
                                                rcmodel.reset_analysis()
