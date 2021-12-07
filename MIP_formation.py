import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon





from gurobipy import Model
from gurobipy import *
from math import *
from gurobi import GRB
model = Model("ppl")


N = 20 # number of vertices
# T = 1 # time horizon

# limits
# - expansion
s_min, s_max = 0, 10    
x_min, x_max = -10, 10
y_min, y_max = -10, 10
# - translation
t_min, t_max = (x_min - x_max), (x_max - x_min)
        
        
# Define the vertices
"If the shape is not zero centered, the scaling equation must be changed!"
vertices = [[0.5, 1], [-0.5, 1], [-0.5, -1], [0.5, -1]]
vertices = np.array(vertices)
# vertices_list = [vertices + shift for shift in np.linspace(-5, 5, 10)]

# Let's shift the obstacle from -5 to 5 with linspace




#
# ---
#
def cs(gamma):
    "Rotation matrix"
    mx = [[cos(gamma), -sin(gamma)],
          [sin(gamma), cos(gamma)]]
    return np.array(mx)
rotation_res = 10
CS = [cs(gamma) for gamma in np.linspace(-math.pi/2, math.pi/2, rotation_res)]

# expansion
s = model.addVars(N, lb = s_min, ub = s_max, name = "s")
# translation
t = model.addVars(N, 2, lb = t_min, ub = t_max, name = "t")
# rotation
# phi = model.addVars(1, lb = s_min, ub = s_max)
# q = model.addVars(4, lb = q_min, ub = q_max, name = "q")


# Collision avoidance constraints
d_obs = 0.08 # self.radious

# for all self.obstacles


# Each vertex should be obs_dx, obs_dy away from the obstacle centered at [0, 0]
obs_center = np.array([0, 1])
obs_center_list = [obs_center + np.array([shift, 0]) for shift in np.linspace(-5, 5, N)]
obs_dx = 0.1
obs_dy = 0.1

#
# --- New: mooving obstacle
#






R = 1e5

rotation_chooser  = model.addVars(N, len(CS), lb = 0, vtype = GRB.BINARY)
for t_idx in range(N):
    for phi_idx in range(len(CS)):
        c = model.addVars(len(CS), 4, lb = 0, vtype = GRB.BINARY, name = 'c')
        for vertex in vertices:
            x, y = vertex
            
            # Rotation
            x_rot = CS[phi_idx][0][0] * x + CS[phi_idx][0][1] * y
            y_rot = CS[phi_idx][1][0] * x + CS[phi_idx][1][1] * y
            
            # Scaling
            x_rot  = x_rot * s[t_idx]
            y_rot  = y_rot * s[t_idx]
            
            # Translation
            x_rot = x_rot + t[t_idx,0]
            y_rot = y_rot + t[t_idx,1]
            
            # Constraints
            model.addConstr(  x_rot - (obs_center_list[t_idx][0] + obs_dx) >=  d_obs - R * c[phi_idx, 1] )
            model.addConstr( -x_rot + (obs_center_list[t_idx][0] - obs_dx) >=  d_obs - R * c[phi_idx, 0] )
            model.addConstr(  y_rot - (obs_center_list[t_idx][1] + obs_dy) >=  d_obs - R * c[phi_idx, 3] )
            model.addConstr( -y_rot + (obs_center_list[t_idx][1] - obs_dy) >=  d_obs - R * c[phi_idx, 2] )
            
            model.addConstr(   quicksum(c[phi_idx, q_sum_idx] for q_sum_idx in range(4)) * rotation_chooser[t_idx, phi_idx] <= 3 )
            # model.addConstr(   c[0] + c[1] + c[2] + c[3] <= 3 )
    model.addConstr(quicksum(rotation_chooser[t_idx,i] for i in range(len(CS))) == 1)
# objective function
J = 0
for t_idx in range(N):
    J += (1 - s[t_idx])**2 + t[t_idx, 0]**2 + t[t_idx, 1]**2
    for phi_idx, gamma_ in enumerate(np.linspace(-math.pi/2, math.pi/2, rotation_res)):
        J += rotation_chooser[t_idx, phi_idx] * gamma_**2 * 0.001


model.setObjective(J, GRB.MINIMIZE)
model.Params.Threads = 1
model.Params.TimeLimit = 100

model.optimize()

sol = model.getVars()
model.getVars()
        
# model.computeIIS()
# model.write("model.ilp")
# sol_
sol_t = [[t[t_idx, i].x for i in range(2)] for t_idx in range(N)]
sol_s = [s[t_idx].x for t_idx in range(N)]
sol_rotation_chooser = [[rotation_chooser[t_idx,phi_idx].x for phi_idx in range(len(CS))] for t_idx in range(N)]


# Let us now plot what we have done :))
fig, ax = plt.subplots()

# Plotting the obstacle for every time instance
for t_idx in range(N):
    center = obs_center_list[t_idx]
    corners = []
    corners += [center + np.array([-obs_dx, -obs_dy])] # bottom-left
    corners += [center + np.array([ obs_dx, -obs_dy])] # bottom-right
    corners += [center + np.array([ obs_dx,  obs_dy])] # top-right
    corners += [center + np.array([-obs_dx,  obs_dy])] # top-left
    corners = np.array(corners)
    corners = np.vstack((corners, corners[0, :]))
    ax.plot(corners[:, 0], corners[:, 1], 'k.') # we need this, otherwise the plot doesn't scale its size well, and the polygons wont be visible

    polygon = Polygon(corners, closed=True, fill=True, fc=(0,0,0,0.1), ec=(0,0,0,1), lw=1, zorder = 1)
    ax.add_patch(polygon)
    plt.show()
    
# Plotting the formation
calc_vertices_all = []
for t_idx in range(N):
    calc_vertices = []
    for vertex in vertices:
        x, y = vertex
        myList = sol_rotation_chooser[t_idx]
        val = next((index for index,value in enumerate(myList) if value != 0), None) # https://stackoverflow.com/questions/19502378/python-find-first-instance-of-non-zero-number-in-list/19502692
        phi_idx = val
        # print(phi_idx)
        # print(CS[phi_idx])
        # if phi_idx != 0:
            # kappa = True
        # Rotation
        x_rot = CS[phi_idx][0][0] * x + CS[phi_idx][0][1] * y
        y_rot = CS[phi_idx][1][0] * x + CS[phi_idx][1][1] * y
        
        # Scaling
        x_rot  = x_rot * sol_s[t_idx]
        y_rot  = y_rot * sol_s[t_idx]
        
        # Translation
        x_rot = x_rot + sol_t[t_idx][0]
        y_rot = y_rot + sol_t[t_idx][1]
        
        ax.plot(x_rot, y_rot, 'b.') 
        
        calc_vertices += [x_rot, y_rot]
    calc_vertices_all += [np.array(calc_vertices)]
            
ax.set_aspect('equal', adjustable='box')
plt.show()































