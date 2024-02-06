import gurobipy as gp
from gurobipy import GRB
import numpy as np

def my_callback(model, where):
    global first_callback
    global x
    if where == GRB.Callback.MIPSOL:
        valx = model.cbGetSolution(model._vars_x)
        valzeta = model.cbGetSolution(model.getVarByName('zeta'))

        if first_callback:
            valzeta = float('-inf')
            first_callback=False

        LL = Stage2par(valx,S,cases)
        z = valx + LL 
        
        if z < model._best_obj: #condition 1
            model._best_obj = z              # Update best solution so far
            model._best_x = valx

            if valzeta < LL: #condition 2
                model.cbLazy(x== 1)
                print("Constr added")

                #actual constraints I would want to add in this simple case: 
                # if valx==1:
                #     model.cbLazy(LL *(x) <= zeta)
                # else:
                #     model.cbLazy(-x*LL + LL<= zeta)   


def Stage2par(valx, S,cases):
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
model.setParam(GRB.Param.LazyConstraints, 1)

x = model.addVar(name='x', lb=0.0, ub=1.0, vtype = GRB.BINARY)  
zeta = model.addVar(name='zeta', lb=0.0, vtype = GRB.CONTINUOUS) 
model.setObjective(x+zeta)

# Set the callback function
model._vars_x = x
model._best_obj = float('inf')
model._best_x = None
model.optimize(my_callback)

if model.status == GRB.OPTIMAL:
    print("Objective value is:", model.objVal)
    print(model._vars_x)
    print(model.getVarByName('x'))
    print(model.getVarByName('zeta'))
elif model.status == GRB.INFEASIBLE:
    print('Problem 2 is infeasible!')

