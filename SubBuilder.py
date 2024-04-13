from gurobipy import *
import gurobipy as gp
from gurobipy import GRB

def sub_builder(model_type, x0, y0, scenariotree, model_builder, inputs, init=0):
    scenario, week_range = scenariotree
    scen, C, start_week  = inputs

    if model_type =="sub":
        sub = gp.Model("Demand_test")
        sub.modelSense = GRB.MINIMIZE
    
        #add vars:
        x1, y1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, Pr1 = model_builder.add_2nd_vars_par(sub, week_range)

        #add constr:
        model_builder.add_gen_constraints(sub, week_range, x1, y1, start_week)
        model_builder.add_2nd_constraints(sub, week_range, x0, y0, x1, y1, Pk1, Pdu1, Pdd1, Pc1, PD1, Pl1, Pp1, Pr1, scen[scenario], C, init)
        expr_sub_obj = model_builder.add_obj_sub(x1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, C, scen[scenario]['p'], week_range)
        sub.setObjective(expr_sub_obj) 
        return sub, x1, y1
    
    else:
        #add vars:
        x1, y1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, Pr1 = model_builder.add_2nd_vars(model_type)

        #add constr and generate objective expression
        expr_sub_obj = gp.LinExpr() #placeholder for the objective
        for s in range(scenario):
            model_builder.add_gen_constraints(model_type, week_range, x1[s], y1[s], start_week)
            model_builder.add_2nd_constraints(model_type, week_range, x0, y0, x1[s], y1[s], Pk1[s], Pdu1[s], Pdd1[s], Pc1[s], PD1[s], Pl1[s], Pp1[s],Pr1[s], scen[s], C)
            
            #build the objective expression 
            obj_expr = model_builder.add_obj_sub(x1[s], Pk1[s], Pdu1[s], Pdd1[s], Pc1[s], Pp1[s], PD1[s], Pl1[s], C, scen[s]['p'], week_range)
            expr_sub_obj.add(obj_expr)

        return model_type, expr_sub_obj, x1, y1


