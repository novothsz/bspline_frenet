import numpy as np


x_min, x_max = -10, 10
y_min, y_max = -10, 10
s_min, s_max = 0, 10    # expansion
q_min, q_max = -1, 1
t_min, t_max = (x_min - x_max), (x_max - x_min)


from gurobipy import Model
from gurobipy import *
from math import *
from gurobi import GRB
model = Model("ppl")


N = 4 # number of vertices
T = 1 # time horizon

# Define the vertices
"If the shape is not zero centered, the scaling equation must be changed!"
vertices = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
vertices = np.array(vertices)

#
# --- First attempt
#

# # expansion
# s = model.addVar(name = "s")
# # translation
# t = model.addVars(3, lb = s_min, ub = s_max, name = "t")
# # rotation
# # phi = model.addVars(1, lb = s_min, ub = s_max)
# q = model.addVars(4, lb = q_min, ub = q_max, name = "q")


# from tmp_quaternion import *


# q0 = model.addVar(name = "q0")
# q2 = model.addVar(name = "q2")
# qu = [q0, 0, q2, 0]
# model.addConstr()
# new_vertices = []
# for vertex in vertices:
#     rotated = qv_mult(qu, tuple(vertex) + (0.0, ))
#     scaled = [s * rot for rot in rotated]
#     new_vertices += [t + s * qv_mult(q, tuple(vertex) + (0.0, ))]



#
# --- Second attempt
#
def cs(gamma):
    mx = [[cos(gamma), -sin(gamma)],
          [sin(gamma), cos(gamma)]]
    return np.array(mx)
n_cs = 9
CS = [cs(gamma) for gamma in np.linspace(-math.pi/2, math.pi/2, n_cs)]

# expansion
s = model.addVar(name = "s")
# translation
t = model.addVars(2, lb = s_min, ub = s_max, name = "t")
# rotation
# phi = model.addVars(1, lb = s_min, ub = s_max)
# q = model.addVars(4, lb = q_min, ub = q_max, name = "q")


# Collision avoidance constraints
d_obs = 0.08 # self.radious

# for all self.obstacles


# Each vertex should be obs_dx, obs_dy away from the obstacle centered at [0, 0]
obs_center = [1, 1]
obs_dx = 0.01
obs_dy = 0.01

N = 2 # time samples
R = 1e5

chooser  = model.addVars(len(CS), lb = 0, vtype = GRB.BINARY)
for j in range(len(CS)):
    for k, vertex in enumerate(vertices):
        x, y = vertex
        
        # Rotation
        x_rot = CS[j][0][0] * x + CS[j][0][1] * y
        y_rot = CS[j][1][0] * x + CS[j][1][1] * y
        
        # Scaling
        x_rot  = x_rot * s
        y_rot  = y_rot * s
        
        # x_rot[0] = x_rot[0] * s
        # x_rot[1] = x_rot[1] * s
        
        # y_rot[0] = y_rot[0] * s
        # y_rot[1] = y_rot[1] * s
        
        # Translation
        x_rot = x_rot + t[0]
        y_rot = y_rot + t[1]
        
        c = model.addVars(4, N, lb = 0, vtype = GRB.BINARY, name = 'c_' + str(j) + 'vertex_' + str(k))
        
        # model.addConstrs(( float( x_rot - (obs_center[0] + obs_dx)) >=  d_obs - R * c[1, i] for i in range(N - 1)  ))
        # model.addConstrs(( float(-x_rot + (obs_center[0] - obs_dx)) >=  d_obs - R * c[0, i] for i in range(N - 1)  ))
        # model.addConstrs(( float( y_rot - (obs_center[1] + obs_dy)) >=  d_obs - R * c[3, i] for i in range(N - 1)  ))
        # model.addConstrs(( float(-y_rot + (obs_center[1] - obs_dy)) >=  d_obs - R * c[2, i] for i in range(N - 1)  ))
        
        model.addConstrs((  x_rot - (obs_center[0] + obs_dx) >=  d_obs - R * c[1, i] for i in range(N - 1)  ))
        model.addConstrs(( -x_rot + (obs_center[0] - obs_dx) >=  d_obs - R * c[0, i] for i in range(N - 1)  ))
        model.addConstrs((  y_rot - (obs_center[1] + obs_dy) >=  d_obs - R * c[3, i] for i in range(N - 1)  ))
        model.addConstrs(( -y_rot + (obs_center[1] - obs_dy) >=  d_obs - R * c[2, i] for i in range(N - 1)  ))
        
        model.addConstrs((   c[0, i] + c[1, i] + c[2, i] + c[3, i] <= 3 for i in range(N-1)   ))
model.addConstr(sum(chooser[i] for i in range(len(CS))) == 1)
# objective function
J = 0
J += (1 - s)**2 + t[0]**2 + t[1]**2
for i, gamma_ in enumerate(np.linspace(-math.pi/2, math.pi/2, n_cs)):
    J += chooser[i] * gamma_ 


model.setObjective(J, GRB.MINIMIZE)
model.Params.Threads = 1
model.Params.TimeLimit = 100

model.optimize()

sol = model.getVars()
model.getVars()
        
# model.computeIIS()
# model.write("model.ilp")
        
sol_t = [t[i].x for i in range(2)]
sol_s = s.x
sol_chooser = [chooser[i].x for i in range(len(chooser))]


        

