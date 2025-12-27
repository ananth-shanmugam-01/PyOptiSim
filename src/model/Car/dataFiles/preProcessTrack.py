import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    # Raw Track Data File Path
    fp = "/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/dataFiles/FSUK_2023.csv"
    df = pd.read_csv(fp)

    # re-calculate x, y, psi from curvature and sLap
    dS = np.gradient(df['sLap'])
    curv = df['curv']
    theta = np.cumsum(dS * curv)
    xi = np.cumsum(dS * np.cos(theta))
    yi = np.cumsum(dS * np.sin(theta))

    # Save processed data to a new CSV file
    processed_fp = "/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/dataFiles/FSUK_2023_processed.csv"
    df_processed = pd.DataFrame({'sLap': df['sLap'],'curv': curv, 'theta': theta, 'xi': xi, 'yi': yi})
    df_processed.to_csv(processed_fp, index=False)

