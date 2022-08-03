import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy import sparse
from scipy.linalg import toeplitz
import pandas as pd


class SmartCharging:

    def __init__(self, numberOfHours, pointsInTime):
        # constructor

        # Points in time
        self.T = pointsInTime 

        # time step
        self.Delta = numberOfHours / pointsInTime 
        
        # m is a fixed value representing the grid mix
        m =  np.array([0.314386043,0.314386043,0.46942724,0.46942724,0.528564477,0.528564477,0.548819785,0.548819785,0.555236551,0.555236551,0.564881968,0.564881968,0.565098308,0.565098308,0.554601255,0.554601255,0.503093435,0.503093435,0.373188927,0.373188927,0.226313347,0.226313347,0.208236585,0.208236585,0.209616451,0.209616451,0.205684279,0.205684279,0.201643555,0.201643555,0.197660412,0.197660412,0.193605807,0.193605807,0.193936573,0.193936573,0.194644814,0.194644814,0.195344672,0.195344672,0.193712718,0.193712718,0.188667377,0.188667377,0.190189329,0.190189329,0.193345994,0.193345994])
        self.m = np.expand_dims(m, axis=1)
        
        # PS is rooftop solar generation at each time step
        self.PS = np.zeros((pointsInTime,1)) 
        
        # power limits from the grid 
        self.P_G_min = 0
        self.P_G_max = 200

        # vehicle charge constraints
        self.P_V_lb = np.zeros((pointsInTime,1))


        # storage values
        self.P_B_min = 0
        self.P_B_max = 0
        self.E_B_min = 0
        self.E_B_max = 0
        self.E_B_T_min = 0
        self.E_B_T_max = 0
        self.E_B_1 = 0

    # methods
    def QP_charging(self, pi, PH, w, arrTime, depTime, initEnergy, evCapacity):

        # process initial EV condition
        self.__processInitialConditions(arrTime, depTime, initEnergy, evCapacity)
        # define inequality constraints
        A, b = self.__buildMatrices(PH)
        # define cost functions
        H, f = self.__buildCost(pi, w)
        # optimize
        x, m = self.__quadProg(H, f, A, b)
        # Refine solution
        if ( self.w[3] == 0):
            x, m = self.__refineSolution(A, b, H, f, x, m)
        
        # returns
        PV_star = x.X[0:self.T]
        PB_star = x.X[self.T:2*self.T]
        EV_star = self.A_V @ np.expand_dims(PV_star,axis=1) + self.B_V * self.E_V_1
        EB_star = self.A_B @ np.expand_dims(PB_star,axis=1) + self.B_B * self.E_B_1
         
        return PV_star, PB_star, EV_star, EB_star

    # helper methods
    
    def __reorder_dates(self, pdSeries):
    
        pdSeries1 = pdSeries[:pdSeries.index.get_loc('08:00')]
        pdSeries2 = pdSeries[pdSeries.index.get_loc('08:00'):]
        pdSeries = pd.concat((pdSeries2, pdSeries1))
        
        return pdSeries
    
    def __processInitialConditions(self, arrivingTime, departureTime, initEnergy, evCapacity):

        # vehicle energy constraints
        self.E_V_max = evCapacity # EV max capacity 
        self.E_V_min = 18

        # fixed value - Vehicle energy at final time
        self.E_V_T_max = self.E_V_max # EV final state constraint 
        self.E_V_T_min = 0.9*self.E_V_max
 
        # EV initial energy 
        self.E_V_1 = initEnergy # EV initial energy 

        # build vehicle max charge
        init_ub = pd.Series(np.zeros(self.T))
        init_ub.index = pd.date_range("00:00", "23:59", freq="30min").strftime('%H:%M')
        init_ub.loc[arrivingTime::] = 8.6
        init_ub.loc[:departureTime] = 8.6
        # reorder to match kartik inputs
        init_ub = self.__reorder_dates(init_ub)
        P_V_ub = np.expand_dims(init_ub.values, axis=1)
        # save
        self.P_V_ub = P_V_ub


    def __refineSolution(self, A, b, H, f, x, m):
        
        # If true, original problem was a linear program. 
        # Choosing epsilon > 0 means accepting a controlled amount of
        # suboptimality. Choosing epsilon = 0 preserves optimality
        epsilon = 0.0
        A = sparse.vstack( (A, sparse.csr_matrix(f).transpose()) )
        b = np.concatenate((np.squeeze(b), (1 + epsilon)*(f.T @ x.X)))
        
        # User selector find an optimal or near-optimal solution that satisfies
        # an additional property:
        selector = 3
        
        if selector == 1:
            # Fastest Charge
            H4 = self.S_V @ self.S_V + self.S_B @ self.S_B
            H = 0*H4
            f = self.f3
            x, m = self.__linProg(f, A, b)
        
        elif selector == 2:
            # Min-norm
            H = np.eye(2*self.T)
            f = np.zeros(2*self.T, 1)
            x, m = self.__quadProg(H, f, A, b) # x'*H*x + f'*x
        
        elif selector == 3:
            # Valley-filling
            F = (self.S_V + self.S_B)
            g = (self.PH - self.PS)
            H = F.T @ F
            f = 2*(F.T @ g)
            x, m = self.__quadProg(H, f, A, b) # x'*H*x + f'*x
        
        return x, m
        

    def __linProg(self, f, A, b):
        """Model related procesing"""
        with gp.Env(empty=True) as env:
            env.setParam('LogToConsole', 0)
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("LP")
                m.Params.OutputFlag = 0
                
                # create variables
                x = m.addMVar(shape=A.get_shape()[1], vtype=GRB.CONTINUOUS, name="x")
                
                # multipy by the coefficients
                m.setObjective(f @ x, GRB.MINIMIZE)
                

                # add inequality constraints
                m.addConstr(A @ x <= np.squeeze(b), name="ineq")
                                     
                # Optimize model
                m.optimize()

        return x, m
    
    def __quadProg(self, H, f, A, b):
        """Model related procesing"""
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("QP")
                m.Params.OutputFlag = 0
                
                # create variables
                x = m.addMVar(shape=A.get_shape()[1], vtype=GRB.CONTINUOUS, name="x")
                                
                # multipy by the coefficients
                m.setObjective(x @ H @ x + np.squeeze(f) @ x, GRB.MINIMIZE)
                
                # add inequality constraints
                m.addConstr(A @ x <= np.squeeze(b), name="ineq")
                                     
                # Optimize model
                m.optimize()

        return x, m

    
    def __buildCost(self, pi, w):
        
        # simplification
        I = np.eye(self.T)
        Z = np.zeros([self.T, self.T])
        
        # definition
        f1 = np.concatenate((pi, pi), axis=0)
        S_V = np.concatenate((I, Z), axis=1)
        S_B = np.concatenate((Z, I), axis=1)
        f2 = 1 - np.concatenate((self.m, self.m), axis=0)
        
        f3 = np.concatenate((np.arange(1,self.T+1,1), np.zeros(self.T)), axis=0)
        f3 = np.expand_dims(f3, axis=1)
        
        H4 = S_V.T @ S_V + S_B.T @ S_B # transpose, dot product.
        
        # All together
        H = w[3] * H4
        f = w[0] *f1 + w[1]*f2 + w[2]*f3
        
        # variables that may be required later
        self.S_V = S_V
        self.S_B = S_B
        self.f3 = f3
        self.H4 = H4
        self.w = w
        
        return H, f

    def __buildMatrices(self, PH):
        '''Transform into Standard Quadratic Program Form'''    

        # E_V[t+1] = a_V*E_V[t] + b_V*P_V[t], so:
        a_V = 1
        b_V = self.Delta

        r_V = a_V ** np.arange(0,self.T,1)  
        r_V = r_V * b_V
        
        A_V = np.tril(toeplitz(r_V))

        B_V = np.expand_dims(a_V ** np.arange(1,self.T+1,1), axis=1)
        
        # E_B[t+1] = a_B*E_V[t] + b_B*P_V[t], so:
        a_B = 1;
        b_B = self.Delta;

        r_B = a_B ** np.arange(0,self.T,1)
        r_B = r_B * b_B
 
        A_B = np.tril(toeplitz(r_B))

        B_B = np.expand_dims(a_B ** np.arange(1,self.T+1,1),axis=1)
        
        # simplification
        One = np.ones([self.T, 1])
        w_V = np.expand_dims(A_V[self.T-1, :],axis=0)
        w_B = np.expand_dims(A_B[self.T-1, :],axis=0)
        I = np.eye(self.T)
        Z = np.zeros([self.T, self.T])
        z = np.zeros([1, self.T])
        
        # Constraints
        A = np.block([[I,     I],
                    [-I,     -I],
                    [I,      Z],
                    [-I,     Z],
                    [Z,      I],
                    [Z,      -I],
                    [A_V,    Z],
                    [-A_V,   Z],
                    [Z,      A_B],
                    [Z,      -A_B],
                    [w_V,    z],
                    [-w_V,   z],
                    [z,      w_B],
                    [z,      -w_B]])
        
        # helper array to workaround concatenation
        E_V_B = [self.E_V_T_max - (a_V**(self.T)) * self.E_V_1, 
               -(self.E_V_T_min - (a_V**(self.T)) * self.E_V_1),
               self.E_B_T_max - (a_B**(self.T)) * self.E_B_1,
               -(self.E_B_T_min - (a_B**(self.T)) * self.E_B_1)]
        E_V_B = np.array(E_V_B)
        E_V_B = np.expand_dims(E_V_B, axis=1)
        
        b = np.concatenate((self.P_G_max * One + self.PS - PH,
                            -(self.P_G_min * One + self.PS - PH),
                            self.P_V_ub,
                            -self.P_V_lb,
                            self.P_B_max * One,
                            -self.P_B_min * One,
                            self.E_V_max * One - B_V * self.E_V_1,
                            -(self.E_V_min * One - B_V * self.E_V_1),
                            self.E_B_max * One - B_B * self.E_B_1,
                            -(self.E_B_min * One - B_B * self.E_B_1),
                            E_V_B), axis=0)
        
        # other methods may require these variables
        self.A_V = A_V
        self.B_V = B_V
        self.A_B = A_B
        self.B_B = B_B
        self.PH = PH

        return sparse.csr_matrix(A), b

#+===========================================================================
def main():
    #LMP
    pi = np.array([0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.16923,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809,0.080809])
    pi = np.expand_dims(pi, axis=1)
    # household demand profile
    PH = np.array([3.476,3.476,3,3,2.388,2.388,2.394,2.394,1.362,1.362,2.358,2.358,2.416,2.416,1.516,1.516,2.364,2.364,1.85,1.85,1.516,1.516,2.036,2.036,2.71,2.71,3.87,3.87,3.642,3.642,1.804,1.804,2.826,2.826,1.332,1.332,1.6,1.6,1.444,1.444,1.364,1.364,1.328,1.328,1.35,1.35,4.854,4.854])
    PH = np.expand_dims(PH, axis=1)
    # user defined weights
    w =  np.array([1.0, 0.0, 0.0, 0.0])
    w = np.expand_dims(w, axis=1)
    #EV initial conditions
    arrTime = "18:00"
    depTime = "08:00"
    initEnergy = 65.6353165560742  # max = 85.5
    # create smart charging object
    charging_obj = SmartCharging(numberOfHours=24, pointsInTime=len(pi)) 
    # optimal EV charging
    PV_star, PB_star, EV_star, EB_star = charging_obj.QP_charging(pi, PH, w, arrTime, depTime, initEnergy) # user specific values
    

#+===========================================================================
if __name__ == "__main__":
    main()

