from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

#Improting constructing classes: input and model (constraints)
from InputCreator import InputCreator, get_demand_gen
from ModelBuilder3 import ModelBuilder
from SubBuilder import sub_builder
from ScenarioBuilder import scenario_builder

def master_builder(model, test_type, solve_type, demand_type, dem_inputs, S_init, Prob_s, C, seedsettings, MIP_value=None):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 2), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    #Load inputs, using test_type
    type_dem, type_wv, horizon, current_week, num_weeks, start_week, ses_cur, wv_cur, xval, yval=  dem_inputs

    input_creator = InputCreator(test_type, start_week, num_weeks, current_week)
    L, J, I, loc_mod, mod_locD, dict_bez, dict_cap,dict_cap_month, dict_verpl,dict_bez_ext, \
        skills_empD, emp_skills, loc_emp, emp_locD, emp_t, t_empD, roost_emp_t, dict_prod_av, dict_prod, \
        dict_costs, dict_k, FTE, closed_days, modalities, locs, num_sessions, T= input_creator.General()
    
    #simulate demand:
    if demand_type =="Gen": 
        mod_demI, wv0, ord0, scen, S, dem_basis = get_demand_gen(dem_inputs, S_init, seedsettings) 
  
    #Model characteristics
    model.modelSense = GRB.MINIMIZE

    #Init modelbuilder nd generate variables for stage 0
    model_builder = ModelBuilder(S, L, J, I, loc_mod, mod_locD, dict_bez, dict_cap,dict_cap_month, dict_verpl,dict_bez_ext, skills_empD, emp_skills, \
                loc_emp, emp_locD, emp_t, t_empD, roost_emp_t, dict_prod_av, dict_prod, dict_costs, dict_k, FTE, closed_days, modalities, locs, dem_basis, num_weeks, num_sessions, T, mod_demI) 
    

    #generate constraints for stage0
    if horizon ==12:
        x0, z0, y0, Po0, Pf0, Pdu0,Pdd0, Pp0, PD0, Pr0 = model_builder.add_1st_vars(model)
        model_builder.add_gen_constraints(model,[0, num_weeks], x0, y0, start_week)
        model_builder.add_1st_constraints(model, x0, y0, z0, Po0, Pf0, Pdu0, Pdd0, PD0, Pr0, wv0, ord0,C)
        expr_master_obj = model_builder.add_obj_master(x0, Po0, Pf0, Pdu0,Pdd0, PD0, Pr0, C)
    elif horizon==4:
        x0, y0, Pk0, Pdu0, Pdd0, Pc0, Pp0, PD0, Pl0, Pr0 = model_builder.add_2nd_vars_par(model, [0, num_weeks])
        model_builder.add_gen_constraints(model, [0, num_weeks], x0, y0, start_week)
        model_builder.add_2nd_constraints(model, [0, num_weeks], xval, yval, x0, y0, Pk0, Pdu0, Pdd0, Pc0, PD0, Pl0, Pp0, Pr0, scen[0], C, 1)
        Pyu0, Pyd0 = model_builder.add_Op_constraints(model, y0, yval)
        expr_master_obj = model_builder.add_obj_op(x0, Pk0, Pdu0, Pdd0, Pc0, Pp0, PD0, Pl0, Pr0, C, Pyu0, Pyd0, [0, num_weeks])
        z0={} 

    inputs = [scen, C, start_week]
    inputsS = [modalities,  dict_prod_av, ord0, wv0]
    # #write objective depending on solve type: 
    if solve_type == 'Tact':
        model.setObjective(expr_master_obj)
        model.Params.MIPGap = MIP_value
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Objective value is:", model.objVal)
            get_results = ModelBuilder.ResultBuilder(test_type, 0, x0, y0, wv0, ord0, scen, wv_cur, [0,4], z0, Po0, Pf0)
            get_results.writtendemand(model_builder)
            get_results.get_sessionsT(model_builder)
            get_results.written_lab(model_builder)
            get_results.fig_schedule_1st(model_builder, f"Masterbuilder_0.png")

            return x0, y0

  
            
            # get_results.writtendemand(model_builder)

        elif model.status == GRB.INFEASIBLE:
            print('Problem is infeasible!')
            model.computeIIS()
            model.write("modelINF.ilp")

            with open("modelINF.ilp", "r") as ilp_file:
                ilp_content = ilp_file.read()
                print(ilp_content)
        return model, x0, y0, z0

    elif solve_type== "Extensive":
        #create subproblems within master:
        scenariotree = [S, [0, num_weeks]]
        model, expr_sub_obj, x1, y1 = sub_builder(model, x0, y0, scenariotree, model_builder, inputs)
        #set objective
        model.setObjective(expr_master_obj + expr_sub_obj)
        model.Params.MIPGap = MIP_value
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Objective value is:", model.objVal)
            # get_results = ModelBuilder.ResultBuilder(test_type,0, x0, y0, wv0, ord0, scen, z0, Po0, Pf0)
            # get_results.fig_schedule_1st(model_builder, f"output_{solve_type}_{test_type}_0.png")
            # get_results.writtendemand(model_builder)

            # get_resultsS = ModelBuilder.ResultBuilder(test_type,1, x1, y1, wv0, ord0, scen)
            # get_resultsS.fig_schedule_2nd(model_builder, f"output_{solve_type}_{test_type}_0.png")  

        elif model.status == GRB.INFEASIBLE:
            print('Problem is infeasible!')
            model.computeIIS()
            model.write("modelINF.ilp")
            

            with open("modelINF.ilp", "r") as ilp_file:
                ilp_content = ilp_file.read()
                print(ilp_content)
        return model, x0, y0, z0
        
    elif solve_type == "LShaped":
        #Add placeholder variable for second stage
        zeta = model.addVar(name='zeta', lb=0.0, vtype = GRB.CONTINUOUS) 

        # Set for the callback function
        model._vars_x0 = x0
        model._vars_y0 = y0
        model._vars_z0 = z0
        model._vars_Pdd0 = Pdd0
        model._vars_Pdu0 = Pdu0
        model._vars_PD0 = PD0
        model._vars_Pr0 = Pr0
        model._best_x0 = None
        model._best_y0 = None
        model._best_z0 = None
        model._best_zeta = None
        model._best_obj = float('inf')
        model._best_fair = None

        if horizon ==12:
            model._vars_Po0 = Po0
            model._vars_Pf0 = Pf0

        if horizon ==4:
            model._vars_Pyu0 = Pyu0
            model._vars_Pyd0 = Pyd0
            model._vars_Pc0 = Pc0
            model._vars_Pl0 = Pl0
            model._vars_Pp0 = Pp0
            model._vars_Pk0 = Pk0

        #Set objective
        model.setObjective(expr_master_obj+zeta)
    
        #calculate Lmin
        goalP = 0
        for s in range(S):
            goal = 0
            for i in mod_demI:
                cap = 0
                cost_cap = 0
                for l in mod_locD[i]:
                    cap += dict_cap[i,l][1] #max
                    cost_cap += dict_cap[i,l][1]*dict_costs[i,l]
                #weekly
                for theta in range(0, num_weeks):
                    if scen[s][i,theta,'wv'] < dem_basis[i] - num_sessions:
                        if scen[s][i,theta, 'o']> cap:
                            goal += cost_cap 
                            goal += (scen[s][i,theta, 'o']- cap-0.25*num_sessions)*C["C1_d"]
                        else: 
                            goal += C["C1"]*(scen[s][i,theta, 'o'] -0.25*num_sessions)*2
                    elif dem_basis[i] - num_sessions <=scen[s][i,theta,'wv'] and scen[s][i,theta,'wv'] < dem_basis[i] +num_sessions:
                        if scen[s][i,theta, 'o']> cap:
                            goal += cost_cap
                            goal += (scen[s][i,theta, 'o']- cap)*C["C1_d"]
                        else: 
                            goal += C["C1"]*scen[s][i,theta, 'o']*2
                    elif dem_basis[i] + num_sessions <=scen[s][i,theta,'wv'] and scen[s][i,theta,'wv'] < dem_basis[i] +0.25*num_sessions:
                        if scen[s][i,theta, 'o']+0.25*num_sessions> cap:
                            goal += cost_cap
                            goal += (scen[s][i,theta, 'o']+ 0.25*num_sessions- cap)*C["C1_d"]
                        else: 
                            goal += C["C1"]*(scen[s][i,theta, 'o']+0.25*num_sessions)*2
                    elif scen[s][i,theta,'wv'] >= dem_basis[i] + 0.5*num_sessions:
                        if scen[s][i,theta, 'o']+0.5*num_sessions> cap:
                            goal += cost_cap
                            goal += (scen[s][i,theta, 'o']+ 0.5*num_sessions- cap)*C["C1_d"]
                        else: 
                            goal += C["C1"]*(scen[s][i,theta, 'o']+0.5*num_sessions)*2



                
                # #monthly 
                # sum_ord = 0
                # if scen[s][i,0,'wv'] in range(dem_basis[i] - num_sessions, dem_basis[i] +num_sessions):
                #     sum_ord = (gp.quicksum(scen[s][i,theta, 'o'] for theta in range(num_weeks))).getValue()
                #     if sum_ord> num_weeks*cap:
                #         goal += cost_cap*num_weeks
                #         goal += (sum_ord- num_weeks*cap)*C["C1_d"]
                #     else: 
                #         goal += C["C1"]*sum_ord*2
                # elif scen[s][i,0,'wv'] in range(dem_basis[i] + num_sessions, dem_basis[i] +2*num_sessions):
                #     if sum_ord+num_sessions>num_weeks*cap:
                #         goal += cost_cap*num_weeks
                #         goal += (sum_ord+ num_sessions- num_weeks*cap)*C["C1_d"]
                #     else: 
                #         goal += C["C1"]*(sum_ord +num_sessions)*2
                # elif scen[s][i,0, 'wv'] > dem_basis[i] + 2*num_sessions:
                #     if sum_ord+2*num_sessions>num_weeks*cap:
                #         goal += cost_cap*num_weeks
                #         goal += (sum_ord+ 2*num_sessions- num_weeks*cap)*C["C1_d"]
                #     else: 
                #         goal += C["C1"]*(sum_ord +2*num_sessions)*2
            goalP += scen[s]['p']*goal
        Lmin = goalP

        return model, x0, y0, z0, zeta, S, model_builder, inputs, inputsS, Lmin
    


















