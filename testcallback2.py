import gurobipy as gp
from gurobipy import GRB
import numpy as np

def my_callback(model, where):
    global first_callback
    global x
    global zeta
    if where == GRB.Callback.MIPSOL:
        valx = model.cbGetSolution(model._vars_x)
        valzeta = model.cbGetSolution(model.getVarByName('zeta'))

        if first_callback:
            valzeta = float('-inf')

        LL = stage2par(valx,S,cases)
        z = valx + LL 
        
        if z < model._best_obj: #condition 1
            if first_callback:
                first_callback=False 
            else: 
                model._best_obj = z            
                model._best_x = valx
                model._best_zeta = valzeta

            if valzeta < LL: #condition 2
                # model.cbLazy(x==1)
                # print("Constr added")

                #actual constraints I would want to add in this simple case: 
                if valx==1:
                    model.cbLazy(LL *(x) <= zeta)
                    print(f"CONS: {LL}*x <= zeta")
                else:
                    model.cbLazy(-x*LL + LL<= zeta)   
                    print(f"CONS: -x{LL} + {LL} <= zeta")

def stage2par(valx, S,cases):
    print("input subproblem:", valx)
    model2 = gp.Model("Demand_test")
    model2.modelSense = GRB.MINIMIZE
    y={}
    for s in range(S):
        y[s]=model2.addVar(name='y#'+str(s), lb=0.0, vtype = GRB.INTEGER) 

    for s in range(S):
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

model.setParam(GRB.Param.PreCrush, 1)
model.setParam(GRB.Param.LazyConstraints, 1)

x = model.addVar(name='x', lb=0.0, ub=1.0, vtype = GRB.BINARY)  
zeta = model.addVar(name='zeta', lb=0.0, vtype = GRB.CONTINUOUS) 
zeta.start = float('inf')
model.setObjective(x+zeta)

# Set the callback function
model._vars_x = x
model._best_obj = float('inf')
model._best_x = None
model._best_zeta = None
model.optimize(my_callback)

if model.status == GRB.OPTIMAL:
    print("Objective value is:", model.objVal, 'or', model._best_obj)
    print("Best", model._best_x, model._best_zeta)
    print(model._vars_x)
    print(model.getVarByName('zeta'))
elif model.status == GRB.INFEASIBLE:
    print('Problem 2 is infeasible!')

