from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
import pickle
import logging
import sys

from timeit import default_timer as timer


#specific Callback classes for Lshaped method
from MyCallback import solve_using_Lshape
from gen_data import gen_data
from MasterBuilder import master_builder

def main():#settings
    logging.basicConfig(filename='main.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s') 
    seconds = timer()   
    #Inits     
    test_type = 2

    locs = [[0,1,2,3,4],[5]]
    solve_type ="LShaped" #Extensive, Tact or LShaped. Tact is myopic method, LShaped uses L-shaped optimization algorithm
    sub_solve_type = "Par" #Parallel or Sequential computation
    demand_type = "Gen" 
    S = 3 #number of scenarios
    td = "M" # M for sampled scenarios, mean for mean scenario 25th for q1 or 75 for q3
    type_wv = "Horizon"
    method = "month" #or "week"
    agg = False #False for PI, if using AG this should be True
    MIP_value = 0.05
    logging.warning("[INIT]: test type: {test_type}, fairness: {num_sessions}, type dem: {type_dem}}, agg: {agg}")
    eps =1
    seedsettings = 1 #for reproducability
    naive= False

    C = {
        "C0": 1,
        "C0_o": 10 * 4,  # overwerken
        "C0_f": 10* 4,   # onderwerken
        "C0_d": 1 ,                # demand beantwoorden
        "C0_D": 1,
        "Cd2" : 2,
        "Cd3" : 4,
        "C0_r": 0.37 * 4,   #roostervrij
        "C1_l": 0.03 * 4,    #ochtend en middag andere shift
        "C1": 1,
        "C1_c": 10 *4,  # cancel sessions
        "C1_d": 3,                   # demand beantwoorden
        "C1_D": 1,               # demand beantwoorden
        "C1_p": 0.07 * 4,    # pannningen
        "C0_y": 10*4
    }

    for i in range(0,16):
        C["C1_k",i] = 0.14*4
    for i in [0,1,2,5]: #echo, ct, mri, angio
        C["C1_k",i] = 0.38*4

    #init
    gen_data(eps+10)
    len_period = 4
    maand = range(2,63,4)#,48,53,57]
    maand_to_eval=[14,18,22,26,30,34,38,42,46,50,54,58,62,0,0,0]#53,57,0,0,0]
    to_eval = [0,0,0] 

    #init RH
    wv_cur = {}
    for theta in range(4*len_period):
        wv_cur[theta,0]= 180#330 #330
        wv_cur[theta,1]= 180 #180
    ses_cur = {}
    for theta in range(0, 16):
        ses_cur[theta]  = {}
        for i in range(2):
            ses_cur[theta][0, 0] = 3#4
            ses_cur[theta][1, 0] = 4.75
    if naive:
        for theta in range(16,63):
            ses_cur[theta]  = {}
            for i in range(2):
                ses_cur[theta][0, 0] = 0.1#4
                ses_cur[theta][1, 0] = 0.1
                ses_cur[theta][0, 1] = 0.1#4
                ses_cur[theta][1, 1] = 0.1
    vals = {}

    #output lists:
    res_gem_w = []
    res_max_w = []
    res_goal = []
    res_objVal = []
    res_fair = []
    res_fte = []
    res_maxcap = []
    res_wv = []
    res_wvP = []
    res_ses_dif = {}
    logging.warning(f"[START] Rolling horizon {td},{S}, {method}")
    for index, current_week in enumerate(maand):
        #update counters
        to_eval[2]=to_eval[1]
        to_eval[1]=to_eval[0]
        to_eval[0]= maand_to_eval[index]
        logging.warning(f"Current week:{current_week}, tactical: {to_eval[0]}({len_period}), operational {to_eval[2]}")

        #calculate work backlog for op, use ses_cur[2]
        if to_eval[2]==0:
            pass
        else:
            if method =="week" and solve_type == "LShaped":
                for c_theta, c_current_week in enumerate(range(current_week, current_week + len_period)):
                    try: 
                        c_start_week = to_eval[2] + c_theta
                        dem_inputs = [td, type_wv, 4, c_current_week, 1, c_start_week, ses_cur, wv_cur, vals[to_eval[2], "x"], vals[to_eval[2], "y"]] 
                        sesIO, gem_w, max_w, goal, wv_cur, fte, maxcap, wv, wv2, ses_dif, fair,  objVal = solve_using_Lshape(test_type, MIP_value, sub_solve_type, demand_type, dem_inputs, agg, S, None, C, seedsettings)
                        ses_cur[c_start_week]= sesIO[c_start_week]
                        #resultaten toevoegen:
                        res_gem_w += gem_w
                        res_max_w += max_w
                        res_goal += goal
                        res_objVal.append(objVal)
                        res_fair += fair
                        res_fte += fte
                        res_maxcap += maxcap
                        res_wv += wv
                        res_wvP += wv2
                        res_ses_dif.update(ses_dif)
                        
                    except Exception as e:
                        logging.error("An error occured:", e)
                        sys.exit(1)
            #operational decision (updates [1])
            elif method == "month" and solve_type == "LShaped":
                try: 
                    dem_inputs = [td, type_wv, 4, current_week, len_period, to_eval[2], ses_cur, wv_cur, vals[to_eval[2], "x"], vals[to_eval[2], "y"]] 
                    sesIO, gem_w, max_w, goal, wv_cur, fte, maxcap, wv, wv2, ses_dif, fair, objVal = solve_using_Lshape(test_type, MIP_value, sub_solve_type, demand_type, dem_inputs, agg, S, None, C, seedsettings)
                    for theta in range(to_eval[2], to_eval[2] +len_period):
                        ses_cur[theta]= sesIO[theta]
                    
                    #resultaten toevoegen:
                    res_gem_w += gem_w
                    res_max_w += max_w
                    res_goal += goal
                    res_objVal.append(objVal)
                    res_fair += fair
                    res_fte += fte
                    res_maxcap += maxcap
                    res_wv += wv
                    res_wvP += wv2
                    res_ses_dif.update(ses_dif)
                    
                except Exception as e:
                    logging.error("An error occured:", e)
                    sys.exit(1)
            elif solve_type == "Tact" and not naive:
                dem_inputs = [td, type_wv, 4, current_week, len_period, to_eval[2], ses_cur, wv_cur, vals[to_eval[2], "x"], vals[to_eval[2], "y"]] 
                model = gp.Model("Demand_test")
                sesIO, gem_w, max_w, goal, wv_cur, fte, maxcap, wv, wv2, ses_dif, fair, objVal = master_builder(model, test_type, "Tact", demand_type, dem_inputs, S, None, C, seedsettings, MIP_value)
                for theta in range(to_eval[2], to_eval[2] +len_period):
                    ses_cur[theta]= sesIO[theta]
                
                #resultaten toevoegen:
                res_gem_w += gem_w
                res_max_w += max_w
                res_goal += goal
                res_objVal.append(objVal)
                #res_fair += fair
                res_fte += fte
                res_maxcap += maxcap
                res_wv += wv
                res_wvP += wv2
                res_ses_dif.update(ses_dif)                       

        #tactical decision 
        if to_eval[0] !=0:
            if solve_type == "LShaped":
                try:  
                    dem_inputs = [td, type_wv, 12, current_week, len_period,to_eval[0], ses_cur, wv_cur, {},{}] 
                    sesIT, xval, yval = solve_using_Lshape(test_type, MIP_value, sub_solve_type, demand_type, dem_inputs, agg, S, None, C, seedsettings)
                    for theta in range(to_eval[0], to_eval[0] +len_period):
                        ses_cur[theta]= sesIT[theta]
                    vals[to_eval[0], "x"]= xval
                    vals[to_eval[0], "y"]= yval
                except Exception as e:
                    logging.error("An error occured:", e)
                    sys.exit(1)
            elif solve_type == "Tact" and not naive:
                dem_inputs = [td, type_wv, 4, current_week, len_period,to_eval[0], ses_cur, wv_cur, {},{}] 
                model = gp.Model("Demand_test")
                sesIT, xval, yval = master_builder(model, test_type, "Tact", demand_type, dem_inputs, S, None, C, seedsettings, MIP_value)
                for theta in range(to_eval[0], to_eval[0] +len_period):
                    ses_cur[theta]= sesIT[theta]
                vals[to_eval[0], "x"]= xval
                vals[to_eval[0], "y"]= yval  

            elif solve_type == "Tact" and naive:    
                dem_inputs = [td, type_wv,4, current_week, len_period,to_eval[0], ses_cur, wv_cur, {},{}] 
                model = gp.Model("Demand_test")
                sesIO, gem_w, max_w, goal, wv_cur, fte, maxcap, wv, wv2, ses_dif, fair, objVal  = master_builder(model, test_type, "Tact", demand_type, dem_inputs, S, None, C, seedsettings, MIP_value, naive)
                for theta in range(to_eval[0], to_eval[0] +len_period):
                    ses_cur[theta]= sesIO[theta]
                
                #resultaten toevoegen:
                res_gem_w += gem_w
                res_max_w += max_w
                res_goal += goal
                res_objVal.append(objVal)
                #res_fair += fair
                res_fte += fte
                res_maxcap += maxcap
                res_wv += wv
                res_wvP += wv2
                res_ses_dif.update(ses_dif)     

        if to_eval[0]+to_eval[1]==0:
            logging.warning("[END] Simulation has ended")
            break
    
    #Export results to a file using pickle
    res = [res_gem_w, res_max_w, res_goal, res_fte, res_objVal, res_maxcap, res_wv, res_wvP, res_ses_dif]

    with open(f'gen_{td}_{S}_{method}.pkl', 'wb') as file:
        pickle.dump(res, file)
    with open(f'ses_{td}_{S}_{method}.pkl', 'wb') as file:
        pickle.dump(ses_cur, file)
    with open(f'fair_{td}_{S}_{method}.pkl', 'wb') as file:
        pickle.dump(res_fair, file)

   
if __name__ == '__main__':
    mp.freeze_support()
    main()



