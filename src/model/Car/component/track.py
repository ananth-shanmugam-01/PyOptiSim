import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

""" Collection of functions to create and load track data for CarModel """

def loadTrackData(CarModel, trackFile: str):
    """ Load Track Data from JSON and update model parameters, until then use the values directly"""

    trackData = pd.read_csv(trackFile)

    params = dict()
    params['track'] = dict()
    params['track']['sLap'] = trackData['sLap'].to_numpy()
    params['track']['curv'] = trackData['curv'].to_numpy()
    params['track']['theta'] = trackData['theta'].to_numpy()
    params['track']['xi'] = trackData['xi'].to_numpy()
    params['track']['yi'] = trackData['yi'].to_numpy()

    CarModel.settings.update(params)

    return CarModel

def createSimpleTrack(CarModel, step_length, straight, turn_length, min_radius):

    max_curv = 1/min_radius

    curv_straight = np.zeros( np.ceil(straight/step_length).astype(int) )
    curv_ascent = np.linspace(0,max_curv, np.ceil(turn_length/(2*step_length)).astype(int) )
    curv_descent = np.linspace(max_curv,0, np.ceil(turn_length/(2*step_length)).astype(int) )

    curv = np.concatenate( (curv_straight, curv_ascent, curv_descent[1:], curv_straight) )

    dS = step_length * np.ones( curv.shape )

    psi = np.cumsum( dS * curv )
    sLap = np.cumsum(dS)
    xi = np.cumsum( dS * np.cos(psi) )
    yi = np.cumsum( dS * np.sin(psi) )

    # Plumb into the outputs
    params = dict()
    params['track'] = dict()
    params['track']['sLap'] = sLap
    params['track']['curv'] = curv
    params['track']['theta'] = psi
    params['track']['xi'] = xi
    params['track']['yi'] = yi

    CarModel.settings.update(params)

    return CarModel

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

