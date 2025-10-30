# Indices / enums used across modules (keep ints for JIT friendliness)

# node_f columns (float)
NODE_DEMAND   = 0
NODE_SERVICE  = 1
NODE_TW_OPEN  = 2
NODE_TW_CLOSE = 3
F_NODE_F      = 4

# node_i columns (int)
NODE_PD_PAIR  = 0
NODE_PD_ROLE  = 1  # 1=pickup, 2=delivery, 0=none
NODE_REQUIRED = 2  # 1=required, 0=optional (unserved allowed)
F_NODE_I      = 3

# veh_f columns (float)
VEH_CAPACITY  = 0
VEH_FIXED_COST= 1
VEH_MAX_DUR   = 2
F_VEH_F       = 3

# routes/state (core floats)
F_LOAD = 0
F_DIST = 1
F_COST = 2
F_VEC0 = 3  # start index for vector dims (not used in this minimal demo)

# time state floats
T_ARR   = 0
T_START = 1
T_LEAVE = 2
T_LATE  = 3
T_SLACK = 4
T_WAIT  = 5
F_TIME_F = 6

# penalty vector indices
P_CAP = 0
P_DUR = 1
P_TW  = 2
P_UNS = 3
F_PEN = 4
