"""
by Jorge
"""

# required for processing
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy import sparse


class SLP_dispatch:

    def __init__(self, Gmax, batt, costs, flags, sen, outDSS, initParams):

        self.Gmax = Gmax            # Max generation
        self.cgn = costs["cgn"]              # Cost of generation
        self.clin = costs["clin"]            # "cost of lines"
        self.cctap = costs["cctap"]            # "cost of changing taps
        self.ccap = costs["ccap"]              # "cost of capacity
        self.cdr = costs["cdr"]     # cost of demand response
        self.Pjk_lim = outDSS["limPjks"]      # thermal limit of the line
        self.v_base = outDSS["v_base"]        # Base voltage of the system in volts
        self.vmin = float(initParams["vmin"]) * 1000     # voltage mi
        self.vmax = float(initParams["vmax"]) * 1000     # voltage max
        self.storage = flags["storage"]
        self.DR = flags["DR"]
        self.reg = flags["reg"]
        
        # preprocess
        self.outDSS = outDSS
        ipf = 1 / sen["pf"]
        self.ipf = ipf.to_frame()

        # correct the PTDF by the penalty factor

        PTDF = self.ipf.values.T * sen["PTDF"].values
        self.PTDF = pd.DataFrame(PTDF, index=sen["PTDF"].index, columns=sen["PTDF"].columns)
        
        # attributes
        self.n = len(self.PTDF.columns)  # number of nodes
        self.m = len(self.PTDF)          # number of lines

        # correct the voltage sensitivity by the penalty factor
        dvdp = self.ipf.values.T * sen["dvdp"].values
        self.dvdp = pd.DataFrame(dvdp, index=sen["dvdp"].index, columns=sen["dvdp"].columns)

        # regulator sensitivities
        self.dvdr = sen["dvdr"]
        self.dpjkdr = sen["dpjkdr"]
        self.numRegs = len(sen["dvdr"].columns)

        # battery attributes
        self.numBatteries = batt['numBatteries']
        self.batIncidence = batt['BatIncidence']
        self.batSizes = batt['BatSizes']
        self.batChargingLimits = batt['BatChargingLimits']
        self.batEfficiencies = batt['BatEfficiencies']
        self.batInitEnergy = batt['BatInitEnergy']
        self.batPenalty = batt['BatPenalty']
        self.ccharbat = batt['ccharbat']
        self.ccapacity = batt['ccapacity']

    # Methods
    def PTDF_SLP_OPF(self):
        # preprocess
        demandProfile = self.outDSS["dfDemand"]
        Pjk_0 = self.outDSS["initPjks"]
        v_0 = self.outDSS["initVolts"]
        Pg_0 = self.outDSS["initPower"]
        PDR_0 = self.outDSS["initDR"]

        # define number of points
        self.pointsInTime = np.size(demandProfile, 1)
        self.demandProfilei = demandProfile.any(axis=1)

        # build equality constraints matrices
        Aeq, beq = self.__buildEquality(demandProfile)
        # build inequality constraints matrices
        A, b = self.__buildInequality(Pjk_0, v_0, Pg_0, PDR_0)

        # build cost function and bounds
        ub, lb, f = self.__buildCostAndBounds(demandProfile)
        # add storage portion
        if self.storage:
            # modify A, Aeq
            Aeq, A = self.__addStorage_A(Aeq, A)
            # modify beq, lb, ub, f
            beq, ub, lb, f = self.__addStorage_rest(beq, ub, lb, f)
            lb = lb.tolist()
            ub = ub.tolist()
        # add portion of regulators to matrices
        if self.reg:
            # modify A, Aeq
            Aeq, A = self.__addReg(Aeq, A)
            # modify beq, lb, ub, f
            beq, ub, lb, f = self.__addReg_rest(beq, ub, lb, f)
        # compute linear program optimization
        x, m, LMP = self.__linprog(f, Aeq, beq, A, b, lb, ub)
        return x, m, LMP

    # helper methods
    def __buildCostAndBounds(self, demandProfile):

        if self.DR:
            # max demand response
            DRmax = np.reshape(demandProfile.values.T, (1, demandProfile.size), order="F")

            #  define upper and lower bounds
            ub = np.concatenate((self.Gmax, DRmax), 1)
            lb = np.zeros((1, 2*self.n*self.pointsInTime))
            # define coeffs
            f = np.concatenate((self.cgn, self.cdr), 1)
        else:
            #  define upper and lower bounds
            ub = self.Gmax
            lb = np.zeros((1, self.n * self.pointsInTime))
            # define coeffs
            f = self.cgn

        return ub, lb, f

    def __buildInequality(self, Pjk_0, v_0, Pg_0, PDR_0):
        """Build inequality constraints"""
        # initial power
        self.Pg_0 = np.reshape(Pg_0.values.T, (1, np.size(Pg_0)), order="F")
        # initial demandResponse
        self.PDR_0 = np.reshape(PDR_0.values.T, (1, np.size(PDR_0)), order="F")
        # for voltage ###
        # define limits
        v_base = np.reshape(self.v_base.values.T, (1, np.size(self.v_base.values)), order="F")
        v_lb = -(self.vmin * v_base)
        v_ub = (self.vmax * v_base)

        # compute matrices
        A_v, b_v = self.__buildSensitivityInequality(self.dvdr, self.dvdp, v_0, v_lb, v_ub)

        # for flows ###
        # define limits
        Pjk_lim = np.reshape(self.Pjk_lim.values.T, (1, np.size(self.Pjk_lim.values)), order="F")
        Pjk_lb = Pjk_lim
        Pjk_ub = Pjk_lim
        # compute matrices
        A_flows, b_flows = self.__buildSensitivityInequality(self.dpjkdr, self.PTDF, Pjk_0, Pjk_lb, Pjk_ub)  # restrict only violating lines: this will be done automatically by gurobi
        # concatenate both contributions
        A = sparse.vstack((A_flows, A_v))
        b = np.concatenate((b_flows, b_v), axis=0)
        return A, b

    def __buildSensitivityInequality(self, dxdr, dxdp, x_0, x_lb, x_ub):
        """for both voltage and flows inequalities the procedure is very similar
            This method seeks to standardize the procedure"""
        dxdp = dxdp.values # remove the dataframe
        dxdr = dxdr.values # remove the dataframe
        # Define A
        if self.DR:
            # define A
            A = np.block([[-dxdp, -dxdp],     # -d/dp * Pg - dv/dp * Pdr
                          [dxdp, dxdp]])     # d/dp * Pg + dv/dp * Pdr

            A = sparse.kron(sparse.csr_matrix(A), sparse.csr_matrix(np.eye(self.pointsInTime)))
        else:
            # define A
            A = np.block([[-dxdp],     # -dv/dp * Pg
                          [dxdp]])     # dv/dp * Pg
            A = sparse.kron(sparse.csr_matrix(A), sparse.csr_matrix(np.eye(self.pointsInTime)))

        # Define b

        # reshape initial value
        x_0 = np.reshape(x_0.values.T, (1, np.size(x_0.values)), order="F")
        r_0 = 16 * np.ones((self.numRegs, self.pointsInTime))
        r_0 = np.reshape(r_0.T, (1, np.size(r_0)), order="F")
        # dxdp @ P0:
        # expand dxdp
        dxdp_kron = np.kron(dxdp, np.eye(self.pointsInTime))
        dxdr_kron = np.kron(dxdr, np.eye(self.pointsInTime))
        aux_dxdp = dxdp_kron @ (self.Pg_0 + self.PDR_0).T
        aux_dxdr = dxdr_kron @ r_0.T
        b = np.concatenate((x_lb.T + x_0.T - aux_dxdp - aux_dxdr,
                            x_ub.T - x_0.T + aux_dxdp + aux_dxdr), axis=0)
        return A, b

    def __buildEquality(self, demandProfile):
        """Build equality constraints"""

        balanceNode = self.__balanceNodes(demandProfile)

        # Define Aeq:

        if self.DR:
            # Aeq (power Balance, Demand Response)
            Aeq = np.concatenate((self.ipf.values.T * balanceNode,
                                  self.ipf.values.T * balanceNode), axis=1)       # Nodal Balance Equations
            Aeq = sparse.kron(sparse.csr_matrix(Aeq), sparse.csr_matrix(np.eye(self.pointsInTime)))  # Expand temporal equations
        else:
            # Aeq (power Balance)
            Aeq = self.ipf.values.T * balanceNode           # power balance equations
            Aeq = sparse.kron(sparse.csr_matrix(Aeq), sparse.csr_matrix(np.eye(self.pointsInTime)))  # Expand temporal equations

        # Define beq:

        # total demand for each hour
        balanceDemand = balanceNode @ demandProfile.values
        beq = np.reshape(balanceDemand.T, (1, np.size(balanceDemand)), order='F')

        return Aeq, beq.T

    def __balanceNodes(self, demandProfile):
        """Method designed to asign phase"""
        balanceNode = np.zeros((3, len(self.PTDF.columns)))
        for n, node in enumerate(self.PTDF.columns):
            phi = int(node.split('.')[1])
            balanceNode[phi - 1, n] = 1
        return balanceNode

    def __addReg(self, Aeq, A):
        """ add reg portion for A's"""
        if self.DR:
            auxZero = np.zeros((self.numRegs * (self.pointsInTime + 2), 2 * self.n * self.pointsInTime))
            Aeq1 = sparse.vstack((Aeq,
                                  sparse.csr_matrix(auxZero)))  # Adding part of batteries eff. equations
        else:
            auxZero = np.zeros((self.numRegs * (self.pointsInTime + 2), self.n * self.pointsInTime))
            Aeq1 = sparse.vstack((Aeq,
                                  sparse.csr_matrix(auxZero))) # Adding part of batteries eff. equations
        # Aeq1 at this moment:
        # columns: Pt1...Ptf, PDRt1...PDRtf
        # rows:    (n)*PointsInTime + numRegs*(PointsInTime+2)
        tapChange = sparse.csr_matrix(np.zeros((Aeq.get_shape()[0],
                                               2 * self.numRegs * self.pointsInTime)))  # considering both pos/neg change
        Aeq2 = sparse.hstack((tapChange,
                              sparse.csr_matrix(np.zeros((Aeq.get_shape()[0], self.numRegs * (self.pointsInTime + 1))))))
        # Aeq2 is zero matrix because reg donot affect energy balance
        # columns: Pt1...Ptf, PDRt1..PDRtf, Tt1...Ttf, Et1...Etf
        # rows:    (n)*PointsInTime + (2 + PointsInTime)*numRegs

        Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0 = self.__regAeq2_aux()
        Aeq2_aux = sparse.hstack((Aeq2_auxP, sparse.csr_matrix(Aeq2_auxE)))
        Aeq2 = sparse.vstack((Aeq2, Aeq2_aux))

        # Reg tap final conditions
        Aeq2_aux2 = np.concatenate((np.zeros((self.numRegs, Aeq2_auxP.get_shape()[1])),
                                    np.flip(Aeq2_auxE0.T, 1), np.flip(np.eye(self.numRegs),
                                    0)), axis=1)
        Aeq2 = sparse.vstack((Aeq2, sparse.csr_matrix(Aeq2_aux2)))
        # Aeq2 finally:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        # Build Aeq matrix
        Aeq = sparse.hstack((Aeq1, Aeq2))
        # Aeq:
        # columns: Pnt1,Pntf-Pjkt1,Pjktf-PscBt1,PscBtf-PsdBt1,PsdBtf-EBt1,EBtf
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries

        #
        # Inequalities
        #
        # build Ain:
        Ain = self.__regAin(A)
        return Aeq, Ain

    def __addStorage_A(self, Aeq, A):
        """Compute the battery portion for A's"""
        # Aeq:
        # columns: Pnt1,Pntf
        # rows:    (n+l)*PointsInTime + numBatteries*(PointsInTime+2)
        if self.DR:
            auxZero = np.zeros((self.numBatteries * (self.pointsInTime + 2), 2 * self.n * self.pointsInTime))
            Aeq1 = sparse.vstack((Aeq,
                                  sparse.csr_matrix(auxZero)))  # Adding part of batteries eff. equations
        else:
            auxZero = np.zeros((self.numBatteries * (self.pointsInTime + 2), self.n * self.pointsInTime))
            Aeq1 = sparse.vstack((Aeq,
                                  sparse.csr_matrix(auxZero))) # Adding part of batteries eff. equations

        ############
        # Equalities
        ############
        # build Aeq2:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf
        # rows:    (n)*PointsInTime
        Aeq2 = self.__storageAeq2()
        # build Aeq2_auxP and Aeq2_auxE:
        # Aeq2_auxP:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf
        # rows:    n*PointsInTime (only nodes with storage connected)
        # Aeq2_auxE:
        # columns: EBt1,EBtf
        # rows:    (PointsInTime + 1)*numBatteries
        Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0 = self.__storageAeq2_aux()
        # Adding Energy Balance and Initial Conditions
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf
        # rows:    n*PointsInTime + (1 + PointsInTime)*numBatteries
        Aeq2_aux = sparse.hstack((Aeq2_auxP, sparse.csr_matrix(Aeq2_auxE)))
        Aeq2 = sparse.vstack((Aeq2, Aeq2_aux))

        # Energy Storage final conditions
        # Aeq2 finally:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq2_aux2 = np.concatenate((np.zeros((self.numBatteries, Aeq2_auxP.get_shape()[1])),
                                    np.flip(Aeq2_auxE0.T, 1), np.flip(np.eye(self.numBatteries),
                                    0)), axis=1)
        Aeq2 = sparse.vstack((Aeq2, sparse.csr_matrix(Aeq2_aux2)))

        # Build Aeq matrix
        # Aeq:
        # columns: Pnt1,Pntf-Pjkt1,Pjktf-PscBt1,PscBtf-PsdBt1,PsdBtf-EBt1,EBtf
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq = sparse.hstack((Aeq1, Aeq2))

        ############
        # Inequalities
        ############
        # build Ain:
        Ain = self.__storageAin(A)
        return Aeq, Ain

    def __addReg_rest(self, beq, ub, lb, f):
        """add storage portion to beq, lb, ub, f"""
        tapInit = 16 * np.ones((1, self.numRegs))  # assume initial taps at zero
        beq = np.concatenate((beq, np.zeros((self.pointsInTime * self.numRegs, 1)),
                              tapInit.T, np.zeros((self.numRegs, 1))), 0)

        # add reg portion to lower bounds
        changeTap = 4 * np.ones((1, self.numRegs))
        regMaxTap = 32 * np.ones((1, self.numRegs))

        lb = np.concatenate((lb.T,     # Generation limits
              np.kron(changeTap, np.zeros((1, self.pointsInTime))).T,  # tap changin limits
              np.kron(changeTap, np.zeros((1, self.pointsInTime))).T,  # tap changin limits
              np.kron(regMaxTap, np.zeros((1, self.pointsInTime))).T,     # Regulator lower bound limit 
              np.zeros((self.numRegs, 1))), 0)                   # Initial capacity limits

        # add reg portion to upper bounds
        ub = np.concatenate((ub.T,                         # Generation & Line limits
              np.kron(changeTap, np.ones((1, self.pointsInTime))).T,    # tap changin limits
              np.kron(changeTap, np.ones((1, self.pointsInTime))).T,    # tap changin limits
              np.kron(regMaxTap, np.ones((1, self.pointsInTime))).T,             # Regulator upper bound 
              regMaxTap.T), 0)                   # Initial capacity limits

        f = np.concatenate((f, self.cctap, self.cctap, self.ccap), 1)  # % x = Pg Pdr Plin Psc Psd E E0 r

        return beq, ub, lb, f

    def __addStorage_rest(self, beq, ub, lb, f):
        """add storage portion to beq, lb, ub, f"""
        # add storage portion to beq
        # Balance Energy, Init & Final Conditions
        beq = np.concatenate((beq, np.zeros((self.pointsInTime * self.numBatteries, 1)),
                              self.batInitEnergy.T, np.zeros((self.numBatteries, 1))), 0)

        # add storage portion to lower bounds
        lb = np.concatenate((lb.T,         # Generation limits
              np.kron(self.batChargingLimits, np.zeros((1, self.pointsInTime))).T,  # Charging limits
              np.kron(self.batChargingLimits, np.zeros((1, self.pointsInTime))).T,  # Discharging limits
              np.kron(self.batSizes, np.zeros((1, self.pointsInTime))).T,           # Battery capacity limits
              np.zeros((self.numBatteries, 1))), 0)                                # Initial capacity limits

        # add storage portion to upper bounds
        ub = np.concatenate((ub.T,                         # Generation & Line limits
              np.kron(self.batChargingLimits, np.ones((1, self.pointsInTime))).T,    # Charging limits
              np.kron(self.batChargingLimits, np.ones((1, self.pointsInTime))).T,    # Discharging limits
              np.kron(self.batSizes,np.ones((1, self.pointsInTime))).T,             # Battery capacity limits
              self.batSizes.T), 0)                                            # Initial capacity limits

        f = np.concatenate((f, self.ccharbat, self.ccapacity),1)  # % x = Pg Pdr Plin Psc Psd E E0

        return beq, ub, lb, f

    def __regAin(self, A1):
        """include reg in inequality constraints"""
        # for the voltage
        A2_v = self.__regSensitivityA(self.dvdr)
        # for the flows
        A2_flows = self.__regSensitivityA(self.dpjkdr)

        # contatenate both contributions
        A2 = sparse.vstack((A2_flows, A2_v))

        # finally concatenate with the original matrix
        Ain = sparse.hstack((A1, A2))
        return Ain

    def __storageAin(self, A1):
        """include storage in inequality constraints"""
        # preprocess
        row, _ = np.where(self.batIncidence == 1)  # get nodes with storage

        # for the voltage
        A2_v = self.__storageSensitivityA(row, self.dvdp)
        # for the flows
        A2_flows = self.__storageSensitivityA(row, self.PTDF)

        # contatenate both contributions
        A2 = sparse.vstack((A2_flows, A2_v))

        # finally concatenate with the original matrix
        Ain = sparse.hstack((A1, A2))
        return Ain

    def __regSensitivityA(self, dxdr):
        dxdr = dxdr.values
        # define A
        # portion related to lower bound
        A_b1_aux1 = sparse.kron(-dxdr, sparse.csr_matrix(np.eye(self.pointsInTime)))
        A_b1 = sparse.hstack((sparse.csr_matrix(np.zeros((dxdr.shape[0]*self.pointsInTime, 2 * self.numRegs * self.pointsInTime))), #delta tap
                             A_b1_aux1,  # -dx/dp * (Etap)
                             sparse.csr_matrix(np.zeros((dxdr.shape[0]*self.pointsInTime, self.numRegs))))) # -dx/dr *(0*Etap0)

        # portion related to upper bound
        A_b2_aux1 = sparse.kron(dxdr, sparse.csr_matrix(np.eye(self.pointsInTime)))
        A_b2 = sparse.hstack((sparse.csr_matrix(np.zeros((dxdr.shape[0]*self.pointsInTime, 2 * self.numRegs * self.pointsInTime))),  # delta tap 
                             A_b2_aux1,  # dx/dp *(Etap)
                             sparse.csr_matrix(np.zeros((dxdr.shape[0]*self.pointsInTime, self.numRegs)))))  # -dx/dr *(0*Etap)

        # concatenate both blocks
        A2 = sparse.vstack((A_b1, A_b2))
        return A2

    def __storageSensitivityA(self, row, dxdp):
        dxdp = dxdp.values

        # portion related to lower bound
        A_b1_aux = np.concatenate((dxdp[:, row], - 1./(self.batPenalty)*dxdp[:, row]), 1)
        A_b1_aux1 = sparse.kron(A_b1_aux, sparse.csr_matrix(np.eye(self.pointsInTime)))
        A_b1 = sparse.hstack((A_b1_aux1,  # dx/dp * (Psc-Psd)
                             sparse.csr_matrix(np.zeros((dxdp.shape[0]*self.pointsInTime, self.numBatteries*(self.pointsInTime + 1)) )) )) # -dx/dp *(0*E)

        # portion related to upper bound
        A_b2_aux = np.concatenate((-dxdp[:, row], 1./(self.batPenalty)*dxdp[:, row]), 1)
        A_b2_aux1 = sparse.kron(A_b2_aux, sparse.csr_matrix(np.eye(self.pointsInTime)))
        A_b2 = sparse.hstack((A_b2_aux1,  # -dx/dp *(Psc-Psd)
                             sparse.csr_matrix(np.zeros((dxdp.shape[0]*self.pointsInTime, self.numBatteries*(self.pointsInTime + 1))))))  # -dx/dp *(0*E)

        # concatenate both blocks
        A2 = sparse.vstack((A_b1, A_b2))

        return A2

    def __storageAeq2(self):
        """auxiliary matrices to include storage in equality constraints"""
        # rows with storage
        row, _ = np.where(self.batIncidence == 1)  # get nodes with storage
        ipf = self.ipf.values[row]
        # Aeq2 (Energy Storage impact on power equation)
        # Impact on power equations
        batIncid_Psc = sparse.kron(sparse.csr_matrix(np.eye(len(ipf))),
                                   sparse.csr_matrix(np.eye(self.pointsInTime)))
        batIncid_Psd = sparse.kron(sparse.csr_matrix(np.eye(len(ipf)) * ipf),
                                   sparse.csr_matrix(np.eye(self.pointsInTime)))
        # Batt penalty
        Aeq2 = sparse.hstack((-batIncid_Psc,
                              batIncid_Psd,
                              sparse.csr_matrix(np.zeros((len(ipf) * self.pointsInTime, self.numBatteries * (self.pointsInTime + 1))))))
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf
        # rows:    n*PointsInTime
        return Aeq2

    def __regAeq2_aux(self):
        """auxiliary matrices to include storage in equality constraints"""
        # preprocess
        tapChange = sparse.kron(sparse.csr_matrix(np.eye(self.numRegs)),
                                sparse.csr_matrix(np.eye(self.pointsInTime)))

        # Tap Balance Equations
        # Aeq2_auxP:
        # columns: RegRt1...RegRtf
        # rows:    numRegs*PointsInTime
        Aeq2_auxP = sparse.hstack((-tapChange,
                                   tapChange))
        Aeq2_auxP = sparse.vstack((Aeq2_auxP,
                                  sparse.csr_matrix(np.zeros((self.numRegs, Aeq2_auxP.get_shape()[1])))))

        # Aeq2_auxE:
        # columns: EBt1...EBtf
        # rows:    (PointsInTime + 1)*numRegs
        Aeq2_auxE = np.eye((self.pointsInTime + 1) * self.numRegs)
        for i in range(self.numRegs):
            init = i * self.pointsInTime
            endit = i * self.pointsInTime + self.pointsInTime
            Aeq2_auxE[init:endit, init:endit] = Aeq2_auxE[init:endit, init:endit] - np.eye(self.pointsInTime, k=-1)
        idx_E = [self.pointsInTime * self.numRegs, self.pointsInTime * self.numRegs]
        idx_E0 = [self.pointsInTime * self.numRegs, (self.pointsInTime + 1) * self.numRegs]

        Aeq2_auxE0 = np.zeros((self.pointsInTime * self.numRegs, self.numRegs))
        c = 0
        s = np.sum(Aeq2_auxE[:idx_E[0], :idx_E[1]], 1)
        for i in range(self.pointsInTime * self.numRegs):
            if s[i] == 1:
                Aeq2_auxE0[i, c] = -1
                c += 1
        Aeq2_auxE[:idx_E0[0], idx_E0[0]:idx_E0[1] + 1] = Aeq2_auxE0
        return Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0

    def __storageAeq2_aux(self):
        """auxiliary matrices to include storage in equality constraints"""
        # preprocess
        batIncid_Psc = sparse.kron(sparse.csr_matrix(np.eye(len(self.batEfficiencies.T)) * self.batEfficiencies),
                                   sparse.csr_matrix(np.eye(self.pointsInTime)))
        batIncid_Psd = sparse.kron(sparse.csr_matrix(np.eye(len(self.batEfficiencies.T)) * 1/self.batEfficiencies),
                                   sparse.csr_matrix(np.eye(self.pointsInTime)))

        # Energy Balance Equations
        # Aeq2_auxP:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf
        # rows:    n*PointsInTime (only nodes with storage connected)
        Aeq2_auxP = sparse.hstack((-batIncid_Psc,
                                   batIncid_Psd))
        Aeq2_auxP = sparse.vstack((Aeq2_auxP,
                                  sparse.csr_matrix(np.zeros((self.numBatteries, Aeq2_auxP.get_shape()[1])))))

        # Aeq2_auxE:
        # columns: EBt1,EBtf
        # rows:    (PointsInTime + 1)*numBatteries
        Aeq2_auxE = np.eye((self.pointsInTime + 1) * self.numBatteries)
        for i in range(self.numBatteries):
            init = i * self.pointsInTime
            endit = i * self.pointsInTime + self.pointsInTime
            Aeq2_auxE[init:endit, init:endit] = Aeq2_auxE[init:endit, init:endit] - np.eye(self.pointsInTime, k=-1)
        idx_E = [self.pointsInTime * self.numBatteries, self.pointsInTime * self.numBatteries]
        idx_E0 = [self.pointsInTime * self.numBatteries, (self.pointsInTime + 1) * self.numBatteries]
        Aeq2_auxE0 = np.zeros((self.pointsInTime * self.numBatteries, self.numBatteries))
        c = 0
        s = np.sum(Aeq2_auxE[:idx_E[0], :idx_E[1]], 1)
        for i in range(self.pointsInTime * self.numBatteries):
            if s[i] == 1:
                Aeq2_auxE0[i, c] = -1
                c += 1
        Aeq2_auxE[:idx_E0[0], idx_E0[0]:idx_E0[1] + 1] = Aeq2_auxE0
        return Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0

    def __linprog(self, f, Aeq, beq, A, b, lb, ub):
        """compute LP optmization using gurobi"""
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("LP1")
                m.Params.OutputFlag = 0
                # create variables
                x = m.addMVar(shape=Aeq.get_shape()[1], lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")
                # multipy by the coefficients
                m.setObjective(f @ x, GRB.MINIMIZE)
                # add equality constraints
                m.addConstr(Aeq @ x == np.squeeze(beq), name="eq")
                # add inequality constraints
                m.addConstr(A @ x <= np.squeeze(b), name="ineq")
                ALMP = sparse.hstack((Aeq.transpose(), A.transpose()))
                # Optimize model
                m.optimize()
                # Compute LMP
                LMP = ALMP @ np.expand_dims(m.Pi, 1)

        return x, m, LMP


# +=================================================================================================
def main():
    print('please run the driver first')


# +=================================================================================================
if __name__ == "__main__":
    main()
