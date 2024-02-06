import gurobipy as gp
from gurobipy import GRB
import numpy as np

#vb:
def callBackSubTourElimination(model, where):
    if where == GRB.Callback.MIPSOL:
        solution = model.cbGetSolution(model._vars)
        selected_edges = extractTours(solution)
        if len(selected_edges) < model._number_cities:
            # add subtour elimination constraint
            model.cbLazy(gp.quicksum(model._vars[edge] for edge in selected_edges) <= len(selected_edges) - 1)


def my_callback(model, where):
    global first_callback
    if where == GRB.Callback.MIPSOL:
        valx = model.cbGetSolution(model._vars_x)
        valzeta = model.cbGetSolution(model.getVarByName('zeta'))

        if first_callback:
            valzeta = -10000 #float('-inf')
            first_callback=False

        #Get L1
        LL = Stage2par(valx,S,cases)

        # compute z
        z = valx + LL #first part moet 1st stage objective without zeta
        print("Report", z, model._best_obj)
        if z < model._best_obj:
            # Update best solution so far
            model._best_obj = z
            model._best_x = valx

            if valzeta < LL:
                model.cbLazy(x==1)
                print("ACCESSED")
        

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
model.optimize(my_callback)

if model.status == GRB.OPTIMAL:
    print("Objective value is:", model.objVal)
    print(model._vars_x)
    print(model.getVarByName('x'))
    print(model.getVarByName('zeta'))
elif model.status == GRB.INFEASIBLE:
    print('Problem 2 is infeasible!')

#The value of θ is set to −∞ and is ignored in the initialization computation.