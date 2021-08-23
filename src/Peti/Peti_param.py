import numpy as np

class DecisionVarX():
    def __init__(self, w_list, g_list, lbg, ubg):
        
        self.g_list = g_list
        self.lbg = lbg
        self.ubg = ubg
        
        self.w_list = w_list
        
        self.w0 = []
        self.r = []
        self.r_dot = []
        self.q = []
        self.omega = []
        self.F = []
        self.tau = []
        
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
        
        idx = list(locate(extract_list, lambda a: a == 'r'))
        self.r = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'r_dot'))
        self.r_dot = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'q'))
        self.q = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'omega'))
        self.omega = [w_opt[x] for x in idx]
        
        idx = list(locate(extract_list, lambda a: a == 'F'))
        self.F = [w_opt[x] for x in idx]        
        
        idx = list(locate(extract_list, lambda a: a == 'tau'))
        self.tau = [w_opt[x] for x in idx]          
        
        return self
    
