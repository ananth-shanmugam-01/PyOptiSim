import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


from src.model.BaseModel import BaseModel
import src.tools.DecisionVariables as DecisionVariables

class RocketLanding(BaseModel):

    def createInitialSolution(self, endPoint, numIntervals):

        mesh = np.linspace(0,endPoint,numIntervals)

        g_mesh = 9.81 * np.ones(len(mesh))
        thrust = 20 # N
        mass = 1 
        h = endPoint/numIntervals

        v_forward = np.zeros(len(mesh)) 
        v_forward[0] = 10 # Starting Velocity

        for i in range(1, len(mesh)):
            v_forward[i] = np.sqrt( np.power(v_forward[i-1],2) + 2 * g_mesh[i-1] * h )

        v_braking = np.zeros(len(mesh)) 
        for i in range(len(mesh)-2,-1,-1):
            v_braking[i] = np.sqrt( np.power(v_braking[i+1],2) + 2 *  ((thrust/mass) - g_mesh[i+1] ) * h )

        v_final = np.minimum(v_forward, v_braking)
        thrust_final = thrust * ( v_final == v_braking)

        # Interpolate to Main Mesh 
        initialSolution = dict()
        initialSolution["velocity"] = PchipInterpolator(mesh, v_final)(self.mesh_points)
        initialSolution["mass"] = 1 * np.ones(len(self.mesh_points)) 
        initialSolution["thrust"] = PchipInterpolator(mesh, thrust_final)(self.mesh_points)

        self.initialSolution = initialSolution

    def createModelFunction(self):

        # States
        # addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        velocity = ca.SX.sym('velocity')
        velocity_init = 10 * np.ones(len(self.mesh_points)) 
        self.states = DecisionVariables.addState(self.states, velocity, 'velocity', 'der_velocity', 10, (0, 100), 4, (10, 0), self.initialSolution["velocity"] )
        
        mass     = ca.SX.sym('mass')
        mass_init = 1 * np.ones(len(self.mesh_points)) 
        self.states = DecisionVariables.addState(self.states, mass, 'mass', 'der_mass', 1, (0, 10), 1, (1, 0), mass_init)
        
        # Controls
        thrust = ca.SX.sym('thrust')
        thrust_init = np.ones(len(self.mesh_points))
        self.controls = DecisionVariables.addControl(self.controls, thrust, 'thrust',10, (0, 20), self.initialSolution["thrust"])
        
        # Parameters
        g = ca.SX.sym('g')
        g_mesh = 9.81 * np.ones(len(self.mesh_points))
        self.parameters = DecisionVariables.addParameter(self.parameters, g, 'g', g_mesh)
        
        # Assemble Model 
        c = 0.05 # Consumption factor
        Sf = 1/velocity
        
        # Model Dynamics
        rhs = ca.SX.sym('rhs', self.states.num_x)
        rhs[0] = Sf * (g - thrust/mass)
        rhs[1] = Sf * (-c * thrust)
        
        # Stage Cost
        cost = Sf

        # Model Function
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, cost, self.path_constraints.sym, self.auxiliary_outputs.sym],['x', 'u', 'g'], ['rhs', 'cost', 'path_constraints', 'auxiliary_outputs'])

    def factory():
        
        endPoint = 50 # [m] Mesh Distance
        numIntervals = 15 # Number of Phases
        
        modelFun = RocketLanding()
        modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
        modelFun.createMesh(endPoint, numIntervals)
        modelFun.createInitialSolution(endPoint, numIntervals)
        modelFun.createModelFunction()
        
        return modelFun
    
    def createResultPlots(SimOut):

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot( SimOut.sLap , SimOut.states["velocity"], '-o')
        ax1.set_title('Velocity Trajectory')

        ax2.plot( SimOut.sLap , SimOut.states["mass"], '-o')
        ax2.set_title('Mass State Trajectory')

        ax3.plot( SimOut.sLap , SimOut.controls["thrust"], '-o')
        ax3.set_title('Thrust Control Trajectory')
        ax3.set_xlabel('Distance [m]')

        plt.show()

if __name__ == "__main__":

    endPoint = 50 # [m] Mesh Distance
    numIntervals = 15 # Number of Phases
    
    modelFun = RocketLanding()
    modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
    modelFun.createMesh(endPoint, numIntervals)
    modelFun.createModelFunction()
    
    # Test Function
    [x_dot, cost] = modelFun.modelFunction([8, 1], 4, 1);
    print(x_dot)
    print(cost)

