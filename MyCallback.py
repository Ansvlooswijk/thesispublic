import gurobipy as gp
from gurobipy import GRB
import numpy as np

# def add_cut(model, LL, L, valx):
#     Sk=[]
#     complement_Sk = []
#     for key,value in valx.items():
#         if value==1:
#             Sk.append(key)
#         elif value==0:
#             complement_Sk.append(key)
#     model.cbLazy((LL - L) * (gp.quicksum(x[m] for m in Sk) - gp.quicksum(x[m] for m in complement_Sk) - len(Sk) + 1) + L <=zeta)
#     #model.addConstr((LL - L) * (gp.quicksum(x[m] for m in Sk) - gp.quicksum(x[m] for m in complement_Sk) - len(Sk) + 1) + L <=zeta)
#     print("Adding optimality cut:")
#     print(f"({LL} - {L}) * (gp.quicksum(x[m] for m in {Sk}) - gp.quicksum(x[m] for m in {complement_Sk}) - {len(Sk)} + 1) + {L} <=zeta)")

# def add_cut(model, LL, L, valx):
#     if valx==1:
#         model.cbLazy(LL *(x) <= zeta)
#         print(f"constr added: ({LL} - {L}) * ({x}) + {L} <= zeta")
#     else:
#         model.cbLazy(-x*LL + LL<= zeta)   
#         print(f"constr added: ({LL} - {L}) * (-{x} + 1) + {L} <= zeta")
    # Sk = [i for i, value in valx.items() if value == 1]
    # complement_Sk = [i for i, value in valx.items() if value == 0]

    # model.cbLazy((LL - L) * (gp.quicksum(x[m] for m in Sk) - gp.quicksum(x[m] for m in complement_Sk) - len(Sk) + 1) + L <= zeta)
    #print("Adding optimality cut:")
    #print(f"({LL} - {L}) * (gp.quicksum(x[m] for m in {Sk}) - gp.quicksum(x[m] for m in {complement_Sk}) - {len(Sk)} + 1) + {L} <= zeta)")



def Lshape(x,zeta):

    def my_callback(model, where):
        global first_callback
        if where == GRB.Callback.MIPSOL:
            print("ACCESSED")
            valx = model.cbGetSolution(model._vars_x)  #current x
            valzeta = model.cbGetSolution(model.getVarByName('zeta'))
            print(valx, valzeta)
            if first_callback:
                valzeta = float('-inf')
                first_callback=False
                model.cbLazy(x>=1)
                print("FIRST")
            #Get L1
            LL = Stage2par(valx,S,cases)
            print("LL", LL)
            # compute zv
            z = valx + LL #first part moet 1st stage objective without zeta
            print(valx, LL, z, model._best_obj)
            
            
            # if z < model._best_obj:
            #     # Update best solution so far
            #     model._best_obj = z
            #     model._best_x = valx

            #     if valzeta < LL:
            #         L=0     
            #         if valx==1:
            #             model.cbLazy(LL *(x) <= zeta)
            #             print(f"constr added: ({LL} - {L}) * ({x}) + {L} <= zeta")
            #             model.update()
            #         else:
            #             model.cbLazy(-x*LL + LL<= zeta)   
            #             print(f"constr added: ({LL} - {L}) * (-{x} + 1) + {L} <= zeta")
            #             model.update()
    return my_callback
    
def Stage2par(valx,S,cases):
    print('input stage2:',valx)
    model2 = gp.Model("Demand_test")
    model2.modelSense = GRB.MINIMIZE
    y={}
    for s in range(S):
        y[s]=model2.addVar(name='y#'+str(s), lb=0.0, vtype = GRB.INTEGER) 

    for s in range(S):
         #xsum = sum(valx.values())
         xsum =valx
         model2.addConstr(y[s]>=np.ceil(cases[s])- xsum)

    model2.setObjective(gp.quicksum(0.75*y[s] for s in range(S)))

    model2.optimize()

    if model2.status == GRB.OPTIMAL:
        return model2.objVal
    elif model2.status == GRB.INFEASIBLE:
        print('Problem 2 is infeasible!')
        model2.computeIIS()
        model2.write("model2.ilp")


#init inputs
first_callback = True
S=2
cases = [1.3, 2.7]
#init model
model = gp.Model("Demand_test")
model.modelSense = GRB.MINIMIZE
model.setParam(GRB.Param.LazyConstraints, 1)

x = model.addVar(name='x', lb=0.0, ub=1.0, vtype = GRB.BINARY)  
zeta = model.addVar(name='zeta', lb=0.0, vtype = GRB.CONTINUOUS) 
#model.addConstr(-x*3.75 + 3.75<= zeta)  
#model.addConstr(zeta ==x)
model.setObjective(x+zeta)

# Set the callback function
model._vars_x = x
model._best_obj = float('inf')
model._best_x = None
model.optimize(Lshape(x,zeta))

if model.status == GRB.OPTIMAL:
    print("Objective value is:", model.objVal)
    print(model._vars_x)
    print(model.getVarByName('x'))
    print(model.getVarByName('zeta'))
elif model.status == GRB.INFEASIBLE:
    print('Problem 2 is infeasible!')

#The value of θ is set to −∞ and is ignored in the initialization computation.