import numpy as np
class ParamValX():
    def __init__(self, P0_list, P0):
        
        
        self.T = []
        self.x0 = []
        self.xf = []
        # For the mooving Frenet-frame
        self.v_s = []
        self.curvature = []
        self.equation_min_p = []
        self.equation_max_p = []
        self.equation_min_q = []
        self.equation_max_q = []
        self.obst = []
        
        self.z_i = []
        self.z_ji = []
        self.lambda_i = []
        self.lambda_ji = []
        
        self.P0_list = P0_list
        self.P0 = P0
        self.assemble()
        
    def assemble(self):       
        """Assembles an array of values which the x optimization solver accepts.
        The array is constructed from the class member values and arranged in a 
        sequence described by assemble_list.

        Parameters
        ----------
        assemble_list : array
            Describes how the values sould be arranged.
            
        Returns
        -------
        P0_assemble : array
            A list of parameter values for the x optimization solver.
        """
        from more_itertools import locate
        assemble_list = self.P0_list
        idx_T = list(locate(assemble_list, lambda a: a == 'T'))
        idx_x0 = list(locate(assemble_list, lambda a: a == 'x0'))
        idx_xf = list(locate(assemble_list, lambda a: a == 'xf'))
        # For the mooving Frenet-frame
        idx_v_s = list(locate(assemble_list, lambda a: a == 'v_s'))
        idx_curvature = list(locate(assemble_list, lambda a: a == 'curvature'))
        idx_equation_min_p = list(locate(assemble_list, lambda a: a == 'equation_min_p'))
        idx_equation_max_p = list(locate(assemble_list, lambda a: a == 'equation_max_p'))
        idx_equation_min_q = list(locate(assemble_list, lambda a: a == 'equation_min_q'))
        idx_equation_max_q = list(locate(assemble_list, lambda a: a == 'equation_max_q'))
        idx_obst = list(locate(assemble_list, lambda a: a == 'obst'))
        
        idx_z_i = list(locate(assemble_list, lambda a: a == 'z_i'))
        idx_z_ji = list(locate(assemble_list, lambda a: a == 'z_ji'))
        idx_lambda_i = list(locate(assemble_list, lambda a: a == 'lambda_i'))
        idx_lambda_ji = list(locate(assemble_list, lambda a: a == 'lambda_ji'))
        
        
        P0_assemble = np.zeros((1, len(self.P0_list)))[0]
        
        # Only if this is not the first iteration. Otherwise: all zeros
        if self.z_i != []:
            for i, idx in enumerate(idx_T):
                P0_assemble[idx] = self.T[i]
                
            for i, idx in enumerate(idx_x0):
                P0_assemble[idx] = self.x0[i]
                
            for i, idx in enumerate(idx_xf):
                P0_assemble[idx] = self.xf[i]
                
            # For the mooving Frenet-frame
            for i, idx in enumerate(idx_v_s):
                P0_assemble[idx] = self.v_s[i]
            for i, idx in enumerate(idx_curvature):
                P0_assemble[idx] = self.curvature[i]
            for i, idx in enumerate(idx_equation_min_p):
                P0_assemble[idx] = self.equation_min_p[i]
            for i, idx in enumerate(idx_equation_max_p):
                P0_assemble[idx] = self.equation_max_p[i]
            for i, idx in enumerate(idx_equation_min_q):
                P0_assemble[idx] = self.equation_min_q[i]
            for i, idx in enumerate(idx_equation_max_q):
                P0_assemble[idx] = self.equation_max_q[i]
            for i, idx in enumerate(idx_obst):
                P0_assemble[idx] = self.obst[i]
                
                
            for i, idx in enumerate(idx_z_i):
                P0_assemble[idx] = self.z_i[i]
                
            for i, idx in enumerate(idx_z_ji):
                P0_assemble[idx] = self.z_ji[i]
                
            for i, idx in enumerate(idx_lambda_i):
                P0_assemble[idx] = self.lambda_i[i]
                
            for i, idx in enumerate(idx_lambda_ji):
                P0_assemble[idx] = self.lambda_ji[i]
                
                
            # flatten = lambda t: [item for sublist in t for item in sublist]
            P0_assemble = P0_assemble.tolist()
            
            return P0_assemble
            
        else:
            self.T = np.zeros((1, len(idx_T))).tolist()[0]
            self.x0 = np.zeros((1, len(idx_x0))).tolist()[0]
            self.xf = np.zeros((1, len(idx_xf))).tolist()[0]
            
            # For the mooving Frenet-frame
            self.v_s = np.zeros((1, len(idx_v_s))).tolist()[0]
            self.curvature = np.zeros((1, len(idx_curvature))).tolist()[0]
            self.equation_min_p = np.zeros((1, len(idx_equation_min_p))).tolist()[0]
            self.equation_max_p = np.zeros((1, len(idx_equation_max_p))).tolist()[0]
            self.equation_min_q = np.zeros((1, len(idx_equation_min_q))).tolist()[0]
            self.equation_max_q = np.zeros((1, len(idx_equation_max_q))).tolist()[0]
            self.obst = np.zeros((1, len(idx_obst))).tolist()[0]
        
            self.z_i = np.zeros((1, len(idx_z_i))).tolist()[0]
            self.z_ji = np.zeros((1, len(idx_z_ji))).tolist()[0]
            self.lambda_i = np.zeros((1, len(idx_lambda_i))).tolist()[0]
            self.lambda_ji = np.zeros((1, len(idx_lambda_ji))).tolist()[0]
            
            return self
            
        
        
        
class ParamValZ():
    def __init__(self, P0_z_list, P0_z):
        
        
        self.y = []
        self.y_j = []
        self.lambda_i = []
        self.lambda_ij = []
        
        
        self.P0_z_list = P0_z_list
        self.P0_z = P0_z
        self.assemble()
        
    def assemble(self):              
        """Assembles an array of values which the z optimization solver accepts.
        The array is constructed from the class member values and arranged in a 
        sequence described by assemble_list.

        Parameters
        ----------
        assemble_list : array
            Describes how the values sould be arranged.
            
        Returns
        -------
        P0_z_assemble : array
            A list of parameter values for the z optimization solver.
        """
        from more_itertools import locate
        assemble_list = self.P0_z_list
        idx_x_i = list(locate(assemble_list, lambda a: a == 'y'))
        idx_x_j = list(locate(assemble_list, lambda a: a == 'y_j'))
        idx_lambda_i = list(locate(assemble_list, lambda a: a == 'lambda_i'))
        idx_lambda_ij = list(locate(assemble_list, lambda a: a == 'lambda_ij'))
        
        P0_z_assemble = np.zeros((1, len(self.P0_z_list)))[0]
        
        # Only if this is not the first iteration. Otherwise: zeros.
        if self.y != []:
            for i, idx in enumerate(idx_x_i):
                P0_z_assemble[idx] = self.y[i]  
                # P0_z_assemble[0, idx] = self.DvX.xk[4 + i]                
                
            for i, idx in enumerate(idx_x_j):
                P0_z_assemble[idx] = self.y_j[i]
                
            for i, idx in enumerate(idx_lambda_i):
                P0_z_assemble[idx] = self.lambda_i[i]
                
            for i, idx in enumerate(idx_lambda_ij):
                P0_z_assemble[idx] = self.lambda_ij[i]            
                
            # flatten = lambda t: [item for sublist in t for item in sublist]
            P0_z_assemble = P0_z_assemble.tolist()
            
            return P0_z_assemble
        
        else:
            self.y = np.zeros((1, len(idx_x_i))).tolist()[0]
            self.y_j = np.zeros((1, len(idx_x_j))).tolist()[0]
            self.lambda_i = np.zeros((1, len(idx_lambda_i))).tolist()[0]
            self.lambda_ij = np.zeros((1, len(idx_lambda_ij))).tolist()[0]
            
        
    
class DecisionVarZ():
    def __init__(self, w_z_list, g_list, lbg, ubg):
        
        self.g_list = g_list
        self.lbg = lbg
        self.ubg = ubg
        
        self.w_z_list = w_z_list
        self.w0_z = []
        self.z_i = []
        self.z_ij = []
    
    def extract(self, solution):       
        """This function is called after the z optimization. It extracts the 
        information from the solution to the z update and saves them as
        member variables. Saved data are: w0_z, z_i, z_ij. When updating the
        parameters, information should be pulled from here, hence this class
        stores the most recent information.

        Parameters
        ----------
        extract_list : list
            List of strings based on what information and in which order is stored
            in the solution. We use it to know from what index we can gather
            what kind of information.
        solution : list
            The solution, from which we have to gather the data.
        """
        flatten = lambda t: [item for sublist in t for item in sublist]
        from more_itertools import locate
        extract_list = self.w_z_list
        
        try:
            w_z_opt = solution['x'].full() 
            w_z_opt = flatten(w_z_opt.tolist())
        except:
            w_z_opt = solution
            
        self.w0_z = w_z_opt
        
        idx = list(locate(extract_list, lambda a: a == 'z_i'))
        self.z_i = [w_z_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'z_ij'))
        self.z_ij = [w_z_opt[x] for x in idx]
        
        return self
    
    def assemble(self):       
        """Assembles an array of values which the z optimization solver accepts.
        The array is constructed from the class member values and arranged in a 
        sequence described by assemble_list

        Parameters
        ----------
        None
            
        Returns
        -------
        DvZ_assemble : array
            A list of parameter values for the x optimization solver.
        """
        from more_itertools import locate
        assemble_list = self.w_z_list
        
    
        idx_z_i = list(locate(assemble_list, lambda a: a == 'z_i'))
        idx_z_ij = list(locate(assemble_list, lambda a: a == 'z_ij'))

        DvZ_assemble = np.zeros((1, len(self.w_z_list)))[0]
        
        # Only if this is not the first iteration. Otherwise: return self.w0, which should contain a good initialization.
        # (this is becase upon instantiating DvX in the Vehicle() class, we feed in a w0 array to the constructor)
        if self.z_i != []:
            for i, idx in enumerate(idx_z_i):
                DvZ_assemble[idx] = self.z_i[i]
                
            for i, idx in enumerate(idx_z_ij):
                DvZ_assemble[idx] = self.z_ij[i]
                
            DvZ_assemble = DvZ_assemble.tolist()
            
            return DvZ_assemble
            
        else:
            return self.w0_z
    
class DecisionVarX():
    def __init__(self, w_list, g_list, lbg, ubg):
        
        self.g_list = g_list
        self.lbg = lbg
        self.ubg = ubg
        
        self.w_list = w_list
        
        self.w0 = []
        self.T = []
        self.y = []
        self.a = []
        self.b = []
        self.d_tau = []
        
    def check_g_fulfilment(self, solution):
        flatten = lambda t: [item for sublist in t for item in sublist]
        from more_itertools import locate
        
        try:
            w_opt = solution['x'].full() 
            w_opt = flatten(w_opt.tolist())  
        except:
            w_opt = solution
            

    def extract(self, solution):       
        """This function is called after the x optimization. It extracts the 
        information from the solution to the x update and saves them as
        member variables. Saved data are: w0, T, x0, xk, ux, uy, a, b, d_tau.
        When updating the parameters, information should be pulled from here,
        hence this class stores the most recent information.

        Parameters
        ----------
        extract_list : list
            List of strings based on what information and in which order is stored
            in the solution. We use it to know from what index we can gather
            what kind of information.
        solution : list
            The solution, from which we have to gather the data.
        """
        flatten = lambda t: [item for sublist in t for item in sublist]
        from more_itertools import locate
        
        try:
            w_opt = solution['x'].full() 
            w_opt = flatten(w_opt.tolist())  
        except:
            w_opt = solution
            
        self.w0 = w_opt
        extract_list = self.w_list
        
        idx = list(locate(extract_list, lambda a: a == 'T'))
        self.T = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'y'))
        self.y = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'a'))
        self.a = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'b'))
        self.b = [w_opt[x] for x in idx]        
        
        idx = list(locate(extract_list, lambda a: a == 'd_tau'))
        self.d_tau = [w_opt[x] for x in idx]          
        
        return self
    
    def assemble(self):       
        """Assembles an array of values which the x optimization solver accepts.
        The array is constructed from the class member values and arranged in a 
        sequence described by assemble_list

        Parameters
        ----------
        None
            
        Returns
        -------
        DvX_assemble : array
            A list of parameter values for the x optimization solver.
        """
        from more_itertools import locate
        assemble_list = self.w_list
        
    
        idx_T = list(locate(assemble_list, lambda a: a == 'T'))
        idx_y = list(locate(assemble_list, lambda a: a == 'y'))
        idx_a = list(locate(assemble_list, lambda a: a == 'a'))
        idx_b = list(locate(assemble_list, lambda a: a == 'b'))
        idx_d_tau = list(locate(assemble_list, lambda a: a == 'd_tau'))

        DvX_assemble = np.zeros((1, len(self.w_list)))[0]
        
        # Only if this is not the first iteration. Otherwise: return self.w0, which should contain a good initialization.
        # (this is becase upon instantiating DvX in the Vehicle() class, we feed in a w0 array to the constructor)
        if self.y != []:
            for i, idx in enumerate(idx_T):
                DvX_assemble[idx] = self.T[i]
                
            for i, idx in enumerate(idx_y):
                DvX_assemble[idx] = self.y[i]
                
            for i, idx in enumerate(idx_a):
                DvX_assemble[idx] = self.a[i]
                
            for i, idx in enumerate(idx_b):
                DvX_assemble[idx] = self.b[i]
                
            for i, idx in enumerate(idx_d_tau):
                DvX_assemble[idx] = self.d_tau[i]
            
            DvX_assemble = DvX_assemble.tolist()
            
            return DvX_assemble
            
        else:
            return self.w0
        