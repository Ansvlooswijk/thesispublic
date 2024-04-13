import gurobipy as gp
from gurobipy import GRB
import numpy as np
import traceback 
import random
from timeit import default_timer as timer
import copy

#set up & importing variables and classes
from ModelBuilder3 import ModelBuilder

import multiprocessing as mp
from multiprocessing import get_context
from functools import partial
import sys
import logging

#set up & importing variables and classes
from MasterBuilder import master_builder
from ScenarioBuilder import scenario_builder
from SubBuilder import sub_builder
   
def solve_using_Lshape(test_type, MIP_value, sub_solve_type, demand_type, dem_inputs, agg, S_init, Prob_s, C, seedsettings):
    seconds = timer()
    type_dem, type_wv, horizon, current_week, num_weeks, start_week, ses_cur, wv_cur, _, _= dem_inputs
    #build master-init
    model = gp.Model("Demand_test")
    model.Params.Threads = 20
    model.Params.TimeLimit=  1800
    model, x0, y0, z0, zeta, S, model_builder, inputs, inputsS, Lmin = master_builder(model, test_type, "LShaped", demand_type, dem_inputs, S_init, Prob_s, C, seedsettings, MIP_value)
    model.Params.MIPGap = MIP_value
    model.setParam(GRB.Param.LazyConstraints, 1)

    global first_callback
    first_callback = True 

    #build month or weekly scenario tree
    scenariotree = scenario_builder(S, agg, num_weeks)

    #register callback
    my_callback = (cb_lshape("cb", x0, y0, z0, zeta, Lmin, sub_solve_type, scenariotree, C, model_builder, inputs, horizon, num_weeks))
    model.optimize(my_callback) 
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write(f"modelINFN_{current_week}_{horizon}.ilp")
        logging.warning(f"INFEASIBLE: {current_week}, {horizon}")
        model.feasRelaxS(2, True, False, True)
        model.optimize(my_callback)

    init_cut = cb_lshape("cut", x0, y0, z0, zeta, Lmin, sub_solve_type, scenariotree, C, model_builder, inputs, horizon, num_weeks)
    model.addConstr(init_cut).setAttr("Lazy", 1)
    #model.addConstr(zeta >= Lmin)
    
    #solve model. 
    #logging.warning(f"[BEGIN] actual sovlve, Lmin:{Lmin}")
    model.optimize(my_callback)
    if model.status == GRB.INFEASIBLE:
        print('Problem 2 is infeasible!')
        logging.warning(f"INFEASIBLE: {current_week}, {horizon}, restart")
        model.feasRelaxS(2, True, False, True)
        model.optimize(my_callback)
    #logging.warning("[END] Solvingproblem using LShape"
    #             "It took %f sec.", timer() - seconds)
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.status == GRB.TIME_LIMIT:
            logging.error("TIMELIMIT exceeded")
        #logging.warning(f"Objective value is: {model.objVal} or, {model._best_obj}")

        #sample true demand:
        if horizon==12:#horizon
            get_results = ModelBuilder.ResultBuilder(test_type, 0, model._best_x0, model._best_y0, inputsS[3], inputsS[2], None, None, [0,num_weeks], model._best_z0, Po=None, Pf=None)
            
            #plotten
            #get_results.fig_schedule_1st(model_builder, f"resRH/vb2stage/1st_PI_T.png", None)

            #RH
            sesI = get_results.get_sessions(model_builder, start_week) #start_week
            return sesI, model._best_x0, model._best_y0
        
        elif horizon==4:
            get_results = ModelBuilder.ResultBuilder(test_type, 1, model._best_x0, model._best_y0, inputsS[3], inputsS[2], inputs[0], wv_cur, [0,num_weeks], z=None, Po=None, Pf=None) #inputs[0]=scen
            sesI = get_results.get_sessions(model_builder, start_week) #start_week
            res_gem_w, res_max_w, res_goal, wv_new, fte, res_maxcap, res_wv, res_wv2, res_ses_dif = get_results.results_op(model_builder, wv_cur, start_week, ses_cur)
            #get_results.fig_schedule_1st(model_builder, f"resRH/vb2stage/1st_PI_O.png", None)
           
            return sesI, res_gem_w, res_max_w, res_goal, wv_new, fte, res_maxcap, res_wv, res_wv2, res_ses_dif, model._best_fair,  model._best_obj, 
        
        else:
            logging.warning("so here")
            return None, None, None, None

def solve_subproblem(s, valx,valy, model_builder, inputs, init, eval=False, test_type = None):
    try: 
        with gp.Env() as env, gp.Model(env=env) as sub:
            try: 
                sub.Params.TimeLimit=  1200
                sub.Params.Threads = 1
                sub,x1,y1 = sub_builder("sub", valx, valy, s, model_builder, inputs, init)
                sub.optimize()
            except Exception as e:
                line_number = traceback.extract_tb(sys.exc_info()[2])[-1][1]
                logging.error(f"Error in solving subproblem {s} (line {line_number}): {e}")
                sys.exit(1)
            if sub.status == GRB.OPTIMAL or sub.status == GRB.TIME_LIMIT:
                if sub.status == GRB.TIME_LIMIT:
                    logging.error(f"TIMELIMIT sub {s}")
                #plot results
                #s_num = s[0]
                #scen =inputs[0]
                #get_results = ModelBuilder.ResultBuilder(test_type, 1, x1,y1, None, None, scen[s_num], None, s[1], z=None, Po=None, Pf=None)
                #get_results.fig_schedule_1st(model_builder, f"resRH/vb2stage/2nd_PI_T_{s_num}.png", s_num)
                #scenario evaluation
                if eval:
                    s_num = s[0]
                    scen =inputs[0]
                    get_results = ModelBuilder.ResultBuilder(test_type, 1, x1,y1, None, None, scen[0], None, s[1], z=None, Po=None, Pf=None)
                    res_gem_w, res_max_w, res_goal = get_results.results_sample(model_builder)
                    #get_results.fig_schedule_1st(model_builder, f"mainScenO_{s_num[0]}.png")
                    return res_gem_w, res_max_w, res_goal, sub.objVal
                    
                else:
                    return sub.objVal
            
            elif sub.status == GRB.INFEASIBLE:
                logging.warning(f"INFEASIBLE: {s}")
                sub.feasRelaxS(2, True, False, True)
                sub.optimize()
                if sub.status == GRB.OPTIMAL:
                    logging.warning("solved the infeasible beast")
                    return sub.objVal

    except Exception as e:
        line_number = traceback.extract_tb(sys.exc_info()[2])[-1][1]
        logging.error(f"Error in solving subproblem {s} (line {line_number}): {e}")
        sys.exit(1)


def cb_lshape(assignment, x0, y0, z0, zeta, Lmin, sub_solve_type, scenariotree, C, model_builder, inputs, horizon, num_weeks):

    def sequential_solve_subproblem(scenariotree, valx, valy, init=0):
        sumres =0
        valx_list = [{**valx} for _ in range(len(scenariotree))]
        valy_list = [{**valy} for _ in range(len(scenariotree))]
        init_list = np.full(len(scenariotree), init)
        for s in range(len(scenariotree)):
            sumres+= solve_subproblem(scenariotree[s], valx_list[s], valy_list[s], model_builder, inputs, init_list[s])
        return sumres

    def parallel_solve_subproblem(scenariotree,valxx,valyy, init=0):
        try: 
            valx_list = [{**valxx} for _ in range(len(scenariotree))]
            valy_list = [{**valyy} for _ in range(len(scenariotree))]
            init_list = np.full(len(scenariotree), init)
            input_subproblem = zip(scenariotree, valx_list, valy_list, [model_builder] * len(scenariotree), [inputs] * len(scenariotree), init_list)

            # Use multiprocessing to solve subproblems in parallel
            logging.info(f"start-up Pool")  
            with get_context("spawn").Pool() as pool:
                results = pool.starmap(solve_subproblem, input_subproblem)
                logging.info(f"results gathered")
                # Close and terminate pools (ensures shutting down if an error occurs)
                pool.close()
                pool.join()

            sumres = 0
            logging.info(f"results of the subproblems: {results}")
            for s in range(len(scenariotree)):
                sumres += results[s]
            return sumres
        except Exception as e:
            logging.error("An error occured:", e)
            sys.exit(1)
    
    def add_cut(LL, Lmin, valx, valy,valz):
        Skx = {key: value for key, value in valx.items() if value == 1.0}
        complement_Skx = {key: value for key, value in valx.items() if value == 0}
        Skz = {key: value for key, value in valz.items() if value == 1.0}
        complement_Skz = {key: value for key, value in valz.items() if value == 0}
        Sky = {key: value for key, value in valy.items() if value == 1.0}
        complement_Sky = {key: value for key, value in valy.items() if value == 0}
        try: 
            expr = (LL - Lmin) * (gp.quicksum(x0[key] for key in Skx) + gp.quicksum(y0[key] for key in Sky) + gp.quicksum(z0[key] for key in Skz)
                                - gp.quicksum(x0[key] for key in complement_Skx) - gp.quicksum(y0[key] for key in complement_Sky) -gp.quicksum(z0[key] for key in complement_Skz)
                                - len(Skx)-len(Sky)-len(Skz) + 1) + Lmin <= zeta
            return expr
        except KeyError: #handle keyerrors
            for key in Sky.copy():  # Using [:] creates a copy of Sky to avoid modifying it while iterating
                if key not in y0:
                    Sky.pop(key)
                    logging.warning(f"removed: Sky: {key}")
            for key in complement_Sky.copy():  # Using [:] creates a copy of Sky to avoid modifying it while iterating
                if key not in y0:
                    complement_Sky.pop(key)
                    logging.warning(f"removed: Sky: {key}")
            
            expr = (LL - Lmin) * (gp.quicksum(x0[key] for key in Skx) + gp.quicksum(y0[key] for key in Sky) + gp.quicksum(z0[key] for key in Skz)
                                - gp.quicksum(x0[key] for key in complement_Skx) - gp.quicksum(y0[key] for key in complement_Sky) -gp.quicksum(z0[key] for key in complement_Skz)
                                - len(Skx)-len(Sky)-len(Skz) + 1) + Lmin <= zeta
            return expr
                    
            
    def callback_inner(model, where):
        global first_callback
        global LL
        global valx
        global valy
        global valz
        if where == GRB.Callback.MIPSOL:
            logging.info(f"Enter callback, first: {first_callback}")
            valx = model.cbGetSolution(model._vars_x0)
            valy = model.cbGetSolution(model._vars_y0)
            valz = model.cbGetSolution(model._vars_z0)
            valzeta = model.cbGetSolution(model.getVarByName('zeta'))
            
            #double check integer solution
            sum_int = sum(valx.values()) + sum(valy.values())
            if abs(sum_int - round(sum_int)) !=0:
                logging.error(f'non integers in optimal soluition')
            else: 
                if horizon==12:
                    valPdd0 = model.cbGetSolution(model._vars_Pdd0)
                    valPdu0 = model.cbGetSolution(model._vars_Pdu0)
                    valPf0 = model.cbGetSolution(model._vars_Pf0)
                    valPo0 = model.cbGetSolution(model._vars_Po0)
                    valPD0 = model.cbGetSolution(model._vars_PD0)
                    valPr0= model.cbGetSolution(model._vars_Pr0)
                elif horizon==4:
                    valPk0 = model.cbGetSolution(model._vars_Pk0)
                    valPdd0 = model.cbGetSolution(model._vars_Pdd0)
                    valPdu0 = model.cbGetSolution(model._vars_Pdu0)
                    valPc0 = model.cbGetSolution(model._vars_Pc0)
                    valPp0 = model.cbGetSolution(model._vars_Pp0)
                    valPD0 = model.cbGetSolution(model._vars_PD0)
                    valPl0 = model.cbGetSolution(model._vars_Pl0)
                    valPr0= model.cbGetSolution(model._vars_Pr0)
                    valPyu0= model.cbGetSolution(model._vars_Pyu0)
                    valPyd0= model.cbGetSolution(model._vars_Pyd0)

                if sub_solve_type=="Par":
                    LL= parallel_solve_subproblem(scenariotree, valx, valy)
                elif sub_solve_type == "Seq":
                    LL = sequential_solve_subproblem(scenariotree, valx, valy)

                if horizon ==12:
                    expr_master_obj = model_builder.add_obj_master(valx, valPo0, valPf0, valPdu0, valPdd0, valPD0, valPr0, C)
                if horizon ==4:
                    expr_master_obj = model_builder.add_obj_op(valx, valPk0, valPdu0, valPdd0, valPc0, valPp0, valPD0, valPl0, valPr0, C, valPyu0, valPyd0, [0,num_weeks])
                B = expr_master_obj + LL
                
                BB= B.getValue()
                if first_callback:
                    first_callback=False 
                    logging.info(f"terminate first round")
                    model.terminate()
                else: 
                    logging.info(f"if BB<best, {BB}, {model._best_obj}")
                    if BB < model._best_obj: #condition 1
                        model._best_obj = BB     
                        model._best_x0 = valx
                        model._best_y0 = valy
                        model._best_z0 = valz
                        model._best_zeta = valzeta
                        if horizon==4:
                            model._best_fair = model_builder.fair_op(valPk0, valPc0, valPp0, valPl0, valPr0, C, valPyu0, valPyd0, num_weeks)
                        if valzeta < LL: #condition 2 
                            model.cbLazy(add_cut(LL,Lmin,valx,valy,valz))
        
    if assignment=="cb":
        return callback_inner
    if assignment=="cut":
       return add_cut(LL, Lmin, valx, valy, valz)
            
            


