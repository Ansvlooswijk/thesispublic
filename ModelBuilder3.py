from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import logging

class ModelBuilder: 
    def __init__(self, S, L, J, I, loc_mod, mod_locD, dict_bez, dict_cap,dict_cap_month, dict_verpl,dict_bez_ext, skills_empD, emp_skills, \
                loc_emp, emp_locD, emp_t, t_empD, roost_emp_t, dict_prod_av, dict_prod, dict_costs, dict_k, FTE, closed_days, modalities, locs, dem_basis, num_weeks, num_sessions, T, mod_dem):
        self.S = S
        self.J = J
        self.I = I
        self.L = L
        self.loc_mod = loc_mod
        self.mod_locD = mod_locD
        self.dict_bez = dict_bez
        self.dict_cap = dict_cap
        self.dict_cap_month = dict_cap_month
        self.dict_verpl = dict_verpl
        self.dict_bez_ext = dict_bez_ext
        self.skills_empD = skills_empD
        self.emp_skills = emp_skills
        self.loc_emp = loc_emp
        self.emp_locD = emp_locD
        self.emp_t = emp_t
        self.t_empD = t_empD
        self.roost_emp_t = roost_emp_t
        self.dict_prod_av = dict_prod_av
        self.dict_prod = dict_prod
        self.dict_costs = dict_costs
        self.dict_k = dict_k
        self.FTE = FTE
        self.closed_days = closed_days
        self.modalities =  modalities
        self.locs = locs
        self.num_weeks = num_weeks
        self.num_sessions = num_sessions
        self.T = T
        self.dem_basis = dem_basis
        self.mod_dem = mod_dem


    def add_1st_vars(self, model):
        x0={} 
        z0={}
        for t in range(self.T):
            for l in range(self.L):
                for i in self.loc_mod[l]:
                    x0[i,t,l]=model.addVar(name='x0#'+str(i)+","+str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)  
                    z0[i,t,l]=model.addVar(name='z0#'+str(i)+","+str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)      
                
        y0={}
        for t in range(self.T):
            for j in self.t_empD[t]:
                for l in self.emp_locD[j]:
                    for i in set(self.emp_skills[j]) & set(self.loc_mod[l]): #common elements
                        y0[i,j,t,l]=model.addVar(name='y0#'+str(i)+","+str(j)+","+str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)
        
        Po0={}
        Pf0={}
        Pp0={}
        for j in range(self.J):
            Po0[j] = model.addVar(name= 'Po0#'+str(j), lb=0.0, vtype= GRB.CONTINUOUS)
            Pf0[j]=  model.addVar(name='Pf0#'+str(j), lb=0.0, vtype = GRB.CONTINUOUS)
            Pp0[j]= model.addVar(name='Pp0#'+str(j), lb=0.0, vtype = GRB.CONTINUOUS)
            
        Pdu0={}
        Pdd0={}
        PD0 = {}
        for i in self.mod_dem:
            Pdu0[i] = model.addVar(name= 'Pdu0#'+str(i), lb=0.0,vtype= GRB.CONTINUOUS)
            Pdd0[i] = model.addVar(name= 'Pdd0#'+str(i), lb=0.0,vtype= GRB.CONTINUOUS)
            PD0[i] = model.addVar(name= 'PD00#'+str(i), lb=0.0, vtype= GRB.CONTINUOUS)
            for theta in range(self.num_weeks):  
                Pdu0[i,theta] = model.addVar(name= 'Pdu0#'+str(i)+","+str(theta),lb=0.0, vtype= GRB.CONTINUOUS)
                Pdd0[i,theta] = model.addVar(name= 'Pdd0#'+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS) 
                PD0[i,theta] = model.addVar(name= 'PD0#'+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS)  

        Pr0={}
        for j in range(self.J):
            for theta in range(self.num_weeks):
                Pr0[j,theta] = model.addVar(name= 'Pr0#'+"_"+str(j)+str(theta), lb = 0.0, vtype= GRB.CONTINUOUS)    


        return x0, z0, y0, Po0, Pf0, Pdu0,Pdd0, Pp0, PD0, Pr0
        
    def add_2nd_vars(self,model):
        x1={}
        Pc1={}
        for s in range(self.S):
            x1[s] = {}
            Pc1[s]= {}
            for theta in range(self.num_weeks):
                for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                    for l in range(self.L):
                        for i in self.loc_mod[l]:
                            x1[s][i,t,l]=model.addVar(name='x1#'+str(s)+"_"+str(i)+","+str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)   
                            Pc1[s][i,t,l]=model.addVar(name='Pc1#'+str(s)+"_"+str(i)+","+str(t)+","+str(l), lb=0.0, vtype = GRB.CONTINUOUS)  
                
        y1={}
        for s in range(self.S):
            y1[s] = {}
            for theta in range(self.num_weeks):
                for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                    for j in self.t_empD[t]:
                        for l in self.emp_locD[j]:
                            for i in set(self.emp_skills[j]) & set(self.loc_mod[l]): #common elements
                                y1[s][i,j,t,l]=model.addVar(name='y1#'+str(s)+"_"+str(i)+","+str(j)+","+ str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)

        Pk1={}
        for s in range(self.S):
            Pk1[s]={}
            for j in range(self.J):
                for i in self.emp_skills[j]:
                    for theta in range(self.num_weeks):
                        Pk1[s][i,j,theta] = model.addVar(name= 'Pk1#'+str(s)+"_"+str(i)+","+str(j)+","+str(theta), lb=0.0, vtype= GRB.CONTINUOUS)
        
        PD1 = {}
        Pdu1={}
        Pdd1={}
        for s in range(self.S):
            Pdu1[s]={}
            Pdd1[s]={}
            PD1[s] = {}
            for i in self.mod_dem:
                Pdu1[s][i] = model.addVar(name= 'Pdu1#'+str(s)+"_"+str(i),lb=0.0, vtype= GRB.CONTINUOUS)
                Pdd1[s][i] = model.addVar(name= 'Pdd1#'+str(s)+"_"+str(i), lb=0.0,vtype= GRB.CONTINUOUS) 
                for theta in range(self.num_weeks):
                    Pdu1[s][i,theta] = model.addVar(name= 'Pdu1#'+str(s)+"_"+str(i)+","+str(theta),lb=0.0, vtype= GRB.CONTINUOUS)
                    Pdd1[s][i,theta] = model.addVar(name= 'Pdd1#'+str(s)+"_"+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS) 
                    PD1[s][i,theta] = model.addVar(name= 'PD1#'+str(s)+"_"+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS)   

        Pp1={}
        for s in range(self.S):
            Pp1[s]={}
            for j in range(self.J):
                Pp1[s][j]= model.addVar(name='Pp1#'+str(j), lb=0.0, vtype = GRB.CONTINUOUS)   

        mornings = list(range(0, self.T, 2))
        Pl1={}
        Pr1 = {}
        for s in range(self.S):
            Pl1[s]={}
            Pr1[s]={}
            for j in range(self.J):
                for theta in range(self.num_weeks):
                    Pr1[s][j,theta] = model.addVar(name= 'Pr1#'+str(j)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS)
            for t in mornings:
                for j in range(self.J):  
                    Pl1[s][j,t] = model.addVar(name= 'Pl1#'+str(s)+"_"+str(j)+","+str(t), lb=0.0,vtype= GRB.CONTINUOUS)  

        return x1, y1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, Pr1
        
    def add_2nd_vars_par(self, model, week_range):
        x1={}
        Pc1={}
        for theta in range(week_range[0], week_range[1]):
            for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                for l in range(self.L):
                    for i in self.loc_mod[l]:
                        x1[i,t,l]=model.addVar(name='x1#'+str(i)+","+str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)   
                        Pc1[i,t,l]=model.addVar(name='Pc1#'+str(i)+","+str(t)+","+str(l), lb=0.0, vtype = GRB.CONTINUOUS)  
                
        y1={}
        for theta in range(week_range[0], week_range[1]):
            for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                for j in self.t_empD[t]:
                    for l in self.emp_locD[j]:
                        for i in set(self.emp_skills[j]) & set(self.loc_mod[l]): #common elements
                            y1[i,j,t,l]=model.addVar(name='y1#'+str(i)+","+str(j)+","+ str(t)+","+str(l), lb=0.0, ub=1.0, vtype = GRB.BINARY)

        Pk1={}
        for j in range(self.J):
            for i in self.emp_skills[j]:
                for theta in range(week_range[0], week_range[1]):
                    Pk1[i,j,theta] = model.addVar(name= 'Pk1#'+str(i)+","+str(j)+","+str(theta), lb=0.0, vtype= GRB.CONTINUOUS)
        
        PD1 = {}
        Pdu1={}
        Pdd1={}
        for i in self.mod_dem:
            Pdu1[i] = model.addVar(name= 'Pdu1#'+str(i),lb=0.0, vtype= GRB.CONTINUOUS)
            Pdd1[i] = model.addVar(name= 'Pdd1#'+str(i), lb=0.0,vtype= GRB.CONTINUOUS) 
            for theta in range(week_range[0], week_range[1]):
                Pdu1[i,theta] = model.addVar(name= 'Pdu1#'+str(i)+","+str(theta),lb=0.0, vtype= GRB.CONTINUOUS)
                Pdd1[i,theta] = model.addVar(name= 'Pdd1#'+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS) 
                PD1[i,theta] = model.addVar(name= 'PD1#'+str(i)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS)    
        
        mornings = list(range(week_range[0]*self.num_sessions, week_range[1]*self.num_sessions, 2))
        Pl1={}
        for t in mornings:
            for j in range(self.J):  
                Pl1[j,t] = model.addVar(name= 'Pl1#'+str(j)+","+str(t), lb=0.0,vtype= GRB.CONTINUOUS)  

        Pp1={}
        Pr1 = {}
        for j in range(self.J):
            Pp1[j]= model.addVar(name='Pp1#'+str(j), lb=0.0, vtype = GRB.CONTINUOUS)  
            for theta in range(week_range[0],week_range[1]):
                Pr1[j,theta] = model.addVar(name= 'Pr1#'+str(j)+","+str(theta), lb=0.0,vtype= GRB.CONTINUOUS)

        return x1, y1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, Pr1
    
    def add_obj_master(self, x0, Po0, Pf0, Pdu0,Pdd0, PD0, Pr0, C):
        expr = (gp.quicksum(gp.quicksum(gp.quicksum(C["C0"]*self.dict_costs[i,l]*x0[i,t,l] for i in set(self.loc_mod[l])&set(self.mod_dem)) for l in range(self.L)) for t in range(self.T))
                        +gp.quicksum(C["C0_o"]*Po0[j]+ C["C0_f"]*Pf0[j] for j in range(self.J))
                        +gp.quicksum(C["C0_d"]*(Pdu0[i]+ Pdd0[i]) +  C["C0_D"]*PD0[i] for i in self.mod_dem)
                        +gp.quicksum(gp.quicksum(1/4*C["C0_d"]*(Pdu0[i,theta]+ Pdd0[i,theta])+ C["C0_D"]*PD0[i,theta] for i in self.mod_dem) for theta in range(self.num_weeks))
                        +gp.quicksum(gp.quicksum(C["C0_r"]*Pr0[j,theta] for j in range(self.J))for theta in range(self.num_weeks)))
        return expr

    def add_obj_sub(self, x1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1,C, p_s, week_range):
        expr =  (p_s*(gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(C["C1"]*self.dict_costs[i,l]*(x1[i,t,l]) for i in set(self.loc_mod[l])&set(self.mod_dem))for l in range(self.L)) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1])) 
        +gp.quicksum(C["C1_p"]*Pp1[j] for j in range(self.J))
        +gp.quicksum(gp.quicksum(C["C1_d"]*(Pdu1[i,theta] + Pdd1[i,theta]) for i in self.mod_dem) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(C["C1_D"]*PD1[i,theta] for i in self.mod_dem) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(gp.quicksum(C["C1_k",i]*Pk1[i,j,theta] for i in self.emp_skills[j]) for j in range(self.J))for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(C["C1_c"]*Pc1[i,t,l] for i in self.loc_mod[l]) for l in range(self.L)) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(C["C1_l"]*Pl1[j,t] for j in range(self.J)) for t in range(week_range[0]*self.num_sessions, week_range[1]*self.num_sessions, 2))))
        return expr 

    def add_obj_op(self, x1, Pk1, Pdu1, Pdd1, Pc1, Pp1, PD1, Pl1, Pr1, C, Pyu, Pyd, week_range):
        expr =  (gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(C["C1"]*self.dict_costs[i,l]*(x1[i,t,l]) for i in set(self.loc_mod[l])&set(self.mod_dem)) for l in range(self.L)) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1])) 
        +gp.quicksum(C["C1_p"]*Pp1[j] for j in range(self.J))
        +gp.quicksum(gp.quicksum(C["C1_d"]*(Pdu1[i,theta] + Pdd1[i,theta]) for i in self.mod_dem) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(C["C1_D"]*PD1[i,theta] for i in self.mod_dem) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(gp.quicksum(C["C1_k",i]*Pk1[i,j,theta] for i in self.emp_skills[j]) for j in range(self.J))for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(C["C1_c"]*Pc1[i,t,l] for i in self.loc_mod[l]) for l in range(self.L)) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(C["C1_l"]*Pl1[j,t] for j in range(self.J)) for t in range(week_range[0]*self.num_sessions, week_range[1]*self.num_sessions, 2))
        +gp.quicksum(gp.quicksum(C["C0_r"]*Pr1[j,theta] for j in range(self.J))for theta in range(week_range[0], week_range[1]))
        +gp.quicksum(gp.quicksum(gp.quicksum(C["C0_y"]*(Pyu[j,t] +Pyd[j,t]) for j in self.t_empD[t]) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1])))
        return expr 
    
    def fair_op(self, Pk1, Pc1, Pp1, Pl1, Pr1, C, Pyu, Pyd, num_weeks):
        penal = []
        for theta in range(num_weeks):
            week_range= [0,0]
            week_range[0]=theta
            week_range[1]=theta+1
            penalweek = (gp.quicksum(C["C1_p"]*Pp1[j] for j in range(self.J))
            +gp.quicksum(gp.quicksum(gp.quicksum(C["C1_k",i]*Pk1[i,j,theta] for i in self.emp_skills[j]) for j in range(self.J))for theta in range(week_range[0], week_range[1]))
            +gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(C["C1_c"]*Pc1[i,t,l] for i in self.loc_mod[l]) for l in range(self.L)) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1]))
            +gp.quicksum(gp.quicksum(C["C1_l"]*Pl1[j,t] for j in range(self.J)) for t in range(week_range[0]*self.num_sessions, week_range[1]*self.num_sessions, 2))
            +gp.quicksum(gp.quicksum(C["C0_r"]*Pr1[j,theta] for j in range(self.J))for theta in range(week_range[0], week_range[1]))
            +gp.quicksum(gp.quicksum(gp.quicksum(C["C0_y"]*(Pyu[j,t] +Pyd[j,t]) for j in self.t_empD[t]) for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0], week_range[1]))).getValue()
            
            penal.append(penalweek)
        return penal

    def add_gen_constraints(self, model, week_range, x, y, start_week):

        #a) minstens zoveel sessies iedere week
        for l in range(self.L):
            for i in set(self.loc_mod[l]): #set(self.loc_emp[l]) &
                for theta in range(week_range[0], week_range[1]): #weekly min and max
                    if self.dict_cap[i,l][0]>0:
                        model.addConstr(gp.quicksum(x[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) >= self.dict_cap[i,l][0]) #b enforces min >> so only use x 

        #b) employee maar op een mod en loc tegelijkertijd
        for theta in range(week_range[0], week_range[1]):
            for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                for j in self.t_empD[t]:
                    model.addConstr(gp.quicksum(gp.quicksum(y[i,j,t,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l])) for l in self.emp_locD[j]) <= 1) #range van kwalifications for personeel j

        #c) Weekly obligated schedules (degene dat altijd open moet zijn is niet meegenomen.
        for (i, l), verpl_list in self.dict_verpl.items(): 
            for theta in range(week_range[0], week_range[1]):
                if start_week in range(31,40) and i==12:
                    pass #in holidays no guidance obligated
                else:
                    verpl_listn = verpl_list + np.ones(len(verpl_list))*theta*self.num_sessions
                    model.addConstr(gp.quicksum(x[i,t,l] for t in verpl_listn) == len(verpl_list)) 

        #d) Ochtend en middag zelfde locatie
        for theta in range(week_range[0],week_range[1]):
            for t in range(theta*self.num_sessions, theta*self.num_sessions+ self.num_sessions, 2):
                for j in range(self.J):
                    for loc in self.locs: 
                        if j in set(self.t_empD[t])&set(self.t_empD[t+1]):
                            if t in self.roost_emp_t[j,theta] or t+1 in self.roost_emp_t[j,theta]:
                                model.addConstr(gp.quicksum(gp.quicksum(y[i,j,t,ll] for i in set(self.emp_skills[j]) & set(self.loc_mod[ll])) for ll in set(loc)&set(self.emp_locD[j]))
                                                                +gp.quicksum(gp.quicksum(y[i,j,t+1,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l])) for l in range(self.L) if l not in loc and l in self.emp_locD[j])
                                                                 <= gp.quicksum(gp.quicksum(y[i,j,t+1,ll] for i in set(self.emp_skills[j]) & set(self.loc_mod[ll])) for ll in set(loc)&set(self.emp_locD[j]))  #loc middag
                                                                    + gp.quicksum(gp.quicksum(y[i,j,t,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l])) for l in set(self.emp_locD[j]))) #overal ochtend
                            else:
                                model.addConstr(gp.quicksum(gp.quicksum(y[i,j,t,l] - y[i,j,t+1,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in set(loc)&set(self.emp_locD[j])) ==0)

        
    def add_1st_constraints(self, model, x, y, z, Po, Pf, Pdu, Pdd, PD, Pr, wv, ord,C):
        
        #e) bezetting 
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for t in range(self.num_sessions):
                    if (i,t,l) in self.dict_bez_ext:
                        for theta in range(self.num_weeks):
                            tt= t + theta*self.num_sessions
                            model.addConstr(gp.quicksum(y[i,j,tt,l] for j in set(self.loc_emp[l]) & set(self.skills_empD[i]) & set(self.t_empD[tt]))-(self.dict_bez[i,l]+ self.dict_bez_ext[i,t,l])*(x[i,tt,l]+z[i,tt,l]) == 0)
                    else:
                        for theta in range(self.num_weeks):
                            tt= t + theta*self.num_sessions
                            model.addConstr(gp.quicksum(y[i,j,tt,l] for j in set(self.loc_emp[l]) & set(self.skills_empD[i]) & set(self.t_empD[tt]))-self.dict_bez[i,l]*(x[i,tt,l]+z[i,tt,l]) == 0)

        #f) contracturen werknemers    
        for j in range(self.J):
            total_bes = (gp.quicksum(len(self.emp_t[j,theta]) for theta in range(self.num_weeks))).getValue() 
            if self.FTE[j]-total_bes > 0:
                model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(y[i,j,t,l] for t in self.emp_t[j,theta])for theta in range(self.num_weeks)) for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])
                             == self.FTE[j] - (self.FTE[j]-total_bes) + Po[j] - Pf[j])
            else: 
                model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(y[i,j,t,l] for t in self.emp_t[j,theta])for theta in range(self.num_weeks)) for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])
                             == self.FTE[j] + Po[j] - Pf[j])
            for theta in range(self.num_weeks): #per week 
                model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(y[i,j,t,l] for t in self.emp_t[j,theta]) for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])
                             <= self.FTE[j]/self.num_weeks + 2)

        # # Punish scheduling on roostervrije dagen
        for j in range(self.J):
            for theta in range(self.num_weeks):
                if len(self.roost_emp_t[j,theta])!=0:
                    model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(y[i,j,t,l] for t in self.roost_emp_t[j,theta])
                                                 for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])
                             == Pr[j, theta])

        #g) alleen reg sessie of flex sessie tegelijkertijd
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for t in range(self.T):
                    model.addConstr(x[i,t,l]+z[i,t,l] <= 1)

        #h)max sessie per week
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for theta in range(self.num_weeks): 
                    if self.dict_cap[i,l][1]!=self.num_sessions:
                        model.addConstr(gp.quicksum(x[i,t,l]+z[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) <= self.dict_cap[i,l][1]) #c

        #i) monthly minimum 
        for i, minmonth in self.dict_cap_month.items(): 
            model.addConstr(gp.quicksum(gp.quicksum(x[i,t,l]+z[i,t,l] for t in range(self.T)) for l in self.mod_locD[i]) >= minmonth) #c
            
        #k demand: dedicated sessions should already cover 40% of the orders per week
        for i in self.mod_dem:
            for theta in range(self.num_weeks):
                model.addConstr(gp.quicksum(gp.quicksum(x[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for l in self.mod_locD[i]) >= np.floor(2/5*ord[i,theta])) #minimal coverage constraint
        
        # #k) DEMAND MONTHLY
        for i in self.mod_dem:
            goalsessies =  4
            goalsessies_int = 2 
            if wv[i] < self.dem_basis[i] - goalsessies_int:
                model.addConstr(C["Cd2"]*Pdu[i] - Pdd[i] == (gp.quicksum(ord[i,theta] for theta in range(self.num_weeks))-2*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x[i,t,l] +z[i,t,l]
                                    for t in range(self.T)) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))

            elif self.dem_basis[i] - goalsessies_int <= wv[i] and  wv[i] < self.dem_basis[i] +goalsessies_int:
                model.addConstr(Pdu[i] - Pdd[i] == gp.quicksum(ord[i,theta] for theta in range(self.num_weeks))*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x[i,t,l] +z[i,t,l]
                                    for t in range(self.T)) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
 
            elif self.dem_basis[i] + goalsessies_int <= wv[i] and wv[i] < self.dem_basis[i] +2*goalsessies_int:
                model.addConstr(Pdu[i] - C["Cd2"]*Pdd[i] == (gp.quicksum(ord[i,theta] for theta in range(self.num_weeks))+ 0.5*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x[i,t,l] +z[i,t,l]
                                    for t in range(self.T)) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
  
            elif wv[i] >= self.dem_basis[i] + 2*goalsessies_int:
                model.addConstr(Pdu[i] -C["Cd3"]*Pdd[i] == (gp.quicksum(ord[i,theta] for theta in range(self.num_weeks))+1*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x[i,t,l] +z[i,t,l]
                                    for t in range(self.T)) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))

        #Panningen constraints: (do not apply to test set)
        try:
            for theta in range(self.num_weeks):
                closed = self.closed_days + np.ones(len(self.closed_days))*self.num_sessions*theta
                model.addConstr(gp.quicksum(x[2,t,5]+ z[2,t,5] for t in closed)==0)    #closing panningen echo
                model.addConstr(gp.quicksum(x[1,t,1]+ z[1,t,1] for t in range(theta*self.num_sessions+1, theta*self.num_sessions+self.num_sessions,2))==0)       #MRI avond (dus middag dicht) 
                model.addConstr(gp.quicksum(x[1,t,2]+ z[1,t,2]for t in range(theta*self.num_sessions, theta*self.num_sessions+ self.num_sessions-2))==0)  #MRI savonds       
        except KeyError:
            pass
            print("Warning: Panningen not properly closed")  

    def add_Op_constraints(self, model, y, yval): 
        Pyu={}
        Pyd={}
        for t in range(self.T):
            for j in self.t_empD[t]:
                Pyu[j,t] = model.addVar(name= 'Py1#'+","+str(j)+","+ str(t),lb=0.0, vtype= GRB.CONTINUOUS)
                Pyd[j,t] = model.addVar(name= 'Py2#' +","+str(j)+","+ str(t), lb=0.0,vtype= GRB.CONTINUOUS) 

        #waar mogelijk personeel hetzelfde
        for theta in range(self.num_weeks):
            for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                for j in self.t_empD[t]:
                    try: 
                        model.addConstr(gp.quicksum(gp.quicksum(y[i,j,t,l] - yval[i,j,t,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j]) + Pyu[j,t]== Pyd[j,t])
                    except KeyError:
                        logging.info(f"key error op constraint: {j}, {t}")
        return Pyu, Pyd
    
    def add_2nd_constraints(self, model, week_range, x0, y0, x1, y1, Pk1, Pdu1, Pdd1, Pc1, PD1, Pl1, Pp1, Pr1, scen, C, init_L=0):

        #e) bezetting zonder z
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for t in range(self.num_sessions):
                    if (i,t,l) in self.dict_bez_ext:
                        for theta in range(week_range[0], week_range[1]):
                            tt= t + theta*self.num_sessions
                            model.addConstr(gp.quicksum(y1[i,j,tt,l] for j in set(self.loc_emp[l]) & set(self.skills_empD[i]) & set(self.t_empD[tt]))-(self.dict_bez[i,l]+ self.dict_bez_ext[i,t,l])*(x1[i,tt,l]) == 0)
                    else:
                        for theta in range(week_range[0], week_range[1]):
                            tt= t + theta*self.num_sessions
                            model.addConstr(gp.quicksum(y1[i,j,tt,l] for j in set(self.loc_emp[l]) & set(self.skills_empD[i]) & set(self.t_empD[tt]))-self.dict_bez[i,l]*(x1[i,tt,l]) == 0)
  
        # #NIEUW l) punish getting rid of sessions
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for theta in range(week_range[0], week_range[1]):
                    for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions):
                        model.addConstr(x0[i,t,l]- x1[i,t,l] <= Pc1[i,t,l]+1-x0[i,t,l]) 

        # #d) Ochtend en middag zelfde mod
        mornings = list(range(week_range[0]*self.num_sessions, week_range[1]*self.num_sessions, 2)) 
        for t in mornings:
            for j in range(self.J):
                if j in set(self.t_empD[t])&set(self.t_empD[t+1]):
                    for i in self.emp_skills[j]:
                        model.addConstr(gp.quicksum(y1[i,j,t,l] - y1[i,j,t+1,l] for l in set(self.mod_locD[i])&set(self.emp_locD[j])) <= Pl1[j,t])

        try:
            for j in self.loc_emp[5]:
                model.addConstr(gp.quicksum(gp.quicksum(y1[i,j,t,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[5]) 
                                                            for t in self.emp_t[j,theta])for theta in range(week_range[0], week_range[1])) + Pp1[j]>= 1)
        except KeyError:
            ("Warning: self.loc_emp[5] does not exist.")

        #NIEUW ) personeel hetzelfde
        if init_L==0:
            for theta in range(week_range[0], week_range[1]):
                for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions): 
                    for j in self.t_empD[t]:
                        model.addConstr(gp.quicksum(gp.quicksum(y0[i,j,t,l] - y1[i,j,t,l] for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])==0)
 
        #h) zonder z
        for l in range(self.L):
            for i in self.loc_mod[l]:
                for theta in range(week_range[0], week_range[1]): #weekly min and max
                    if self.dict_cap[i,l][1]!=self.num_sessions:
                        model.addConstr(gp.quicksum(x1[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) <= self.dict_cap[i,l][1]) #c
                    
   
        #j per week (voorkeur shifts)
        for j in range(self.J):
            for theta in range(week_range[0], week_range[1]): #weekly min and max
                week = range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)
                for _, values in self.dict_k.items():
                    for i in set(values["i"])&set(self.emp_skills[j]):
                        if type(i)==list:
                            i=i[0]
                        model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(y1[ii,j,t,l] for l in set(self.mod_locD[ii]) & set(self.emp_locD[j]))for ii in set(values["i"])&set(self.emp_skills[j])) 
                                            for t in set(week)&set(self.emp_t[j,theta])) + Pk1[i,j,theta]>= values["KeepSkill"] ) #range van kwalifications for personeel j
        #demand week
        for i in self.mod_dem:
            goalsessies = 4
            goalsessies_int = 2
            for theta in range(week_range[0], week_range[1]):
                week = range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)
                if scen[i,theta,'wv'] < self.dem_basis[i] - goalsessies_int:
                    model.addConstr(C["Cd2"]*Pdu1[i,theta] - Pdd1[i,theta] == (scen[i,theta, 'o']-0.5*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x1[i,t,l] 
                                        for t in week) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
                    model.addConstr(Pdd1[i,theta] <= 0.25*PD1[i,theta])
                elif self.dem_basis[i] - goalsessies_int <=scen[i,theta,'wv'] and scen[i,theta,'wv'] < self.dem_basis[i] +goalsessies_int:
                    model.addConstr(Pdu1[i,theta] - Pdd1[i,theta] == scen[i,theta, 'o']*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x1[i,t,l] 
                                        for t in week) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
                    model.addConstr(Pdu1[i,theta] + Pdd1[i,theta]-goalsessies/4 <= PD1[i,theta])
                elif self.dem_basis[i] + goalsessies_int <=scen[i,theta,'wv'] and scen[i,theta,'wv'] < self.dem_basis[i] +2*goalsessies_int:
                    model.addConstr(Pdu1[i,theta] - C["Cd2"]*Pdd1[i,theta] == (scen[i,theta, 'o']+1/8*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x1[i,t,l] 
                                        for t in week) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
                    model.addConstr(Pdu1[i,theta]-goalsessies/4 <= 0.5*PD1[i,theta])
                elif scen[i,theta,'wv'] >= self.dem_basis[i] + 2*goalsessies_int:
                    model.addConstr(Pdu1[i,theta] - C["Cd3"]*Pdd1[i,theta] == (scen[i,theta, 'o'] +0.25*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*gp.quicksum(x1[i,t,l] 
                                        for t in week) for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
                    model.addConstr(Pdu1[i,theta]  <= 0.25*PD1[i,theta])

        # for AG: 
        # for i in self.mod_dem:
        #     goalsessies = self.dem_basis[i]/2
        #     goalsessies_int = self.dem_basis[i]/2*0.75
        #     if scen[i,theta,'wv'] < self.dem_basis[i] - goalsessies_int:
        #         model.addConstr(C["Cd2"]*Pdu1[i] - Pdd1[i] == (gp.quicksum(scen[i,theta, 'o'] for theta in range(week_range[0],week_range[1]))-2*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*
        #                                     gp.quicksum(gp.quicksum(x1[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0],week_range[1]))
        #                                                  for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
        #         #model.addConstr(Pdd[i]  <= 0.25*PD[i])
        #     elif self.dem_basis[i] - goalsessies_int <=scen[i,theta,'wv'] and scen[i,theta,'wv'] < self.dem_basis[i] +goalsessies_int:
        #         model.addConstr(Pdu1[i] - Pdd1[i] == gp.quicksum(scen[i,theta, 'o'] for theta in range(week_range[0],week_range[1]))*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*
        #                                     gp.quicksum(gp.quicksum(x1[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0],week_range[1]))
        #                                                  for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
        #         #model.addConstr(Pdu[i] + Pdd[i]-goalsessies <= PD[i])
        #     elif self.dem_basis[i] + goalsessies_int <=scen[i,theta,'wv'] and scen[i,theta,'wv'] < self.dem_basis[i] +2*goalsessies_int:
        #         model.addConstr(Pdu1[i] - C["Cd2"]*Pdd1[i] == (gp.quicksum(scen[i,theta, 'o'] for theta in range(week_range[0],week_range[1]))+ 0.5*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*
        #                                     gp.quicksum(gp.quicksum(x1[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0],week_range[1]))
        #                                                  for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
        #         #model.addConstr(Pdu[i]-goalsessies <= 0.5*PD[i])
        #     elif scen[i,theta,'wv'] >= self.dem_basis[i] + 2*goalsessies_int:
        #         model.addConstr(Pdu1[i] -C["Cd3"]*Pdd1[i] == (gp.quicksum(scen[i,theta, 'o'] for theta in range(week_range[0],week_range[1]))+1*goalsessies)*self.dict_prod_av[i] - gp.quicksum(gp.quicksum(self.dict_prod[i,l]*
        #                                     gp.quicksum(gp.quicksum(x1[i,t,l] for t in range(theta*self.num_sessions, theta*self.num_sessions + self.num_sessions)) for theta in range(week_range[0],week_range[1]))
        #                                                  for l in set(self.mod_locD[i]) &set(loc)) for loc in self.locs))
        #        #model.addConstr(Pdu[i]  <= 0.25*PD[i])  
               
        #f) contracturen werknemers  
        nw = len(range(week_range[0], week_range[1]))  
        if init_L==1:
            for j in range(self.J):
                for theta in range(week_range[0], week_range[1]): #per week 
                    model.addConstr(gp.quicksum(gp.quicksum(gp.quicksum(y1[i,j,t,l] for t in self.emp_t[j,theta]) for i in set(self.emp_skills[j]) & set(self.loc_mod[l]))for l in self.emp_locD[j])
                                <= self.FTE[j]/nw +1)       
        # #i) monthly minimum 
            for i, minmonth in self.dict_cap_month.items(): 
                model.addConstr(gp.quicksum(gp.quicksum(x1[i,t,l] for t in range(self.T)) for l in self.mod_locD[i]) >= np.floor(minmonth/nw)) 
                    
            
              # weird Panningen constraints:
        try:
            for theta in range(week_range[0], week_range[1]):
                closed = self.closed_days + np.ones(len(self.closed_days))*self.num_sessions*theta
                model.addConstr(gp.quicksum(x1[2,t,5] for t in closed)==0)    #closing panningen echo
                model.addConstr(gp.quicksum(x1[1,t,1] for t in range(theta*self.num_sessions+1,theta*self.num_sessions+self.num_sessions,2))==0)       #MRI avond (dus middag dicht) 
                model.addConstr(gp.quicksum(x1[1,t,2] for t in range(theta*self.num_sessions, theta*self.num_sessions+ self.num_sessions-2))==0)  #MRI savonds   
        except KeyError:
            print("Warning: Panningen not properly closed/MRI not closed")  

    class ResultBuilder(): #nested class 

        def __init__(self, TestType, stage, x, y,wv, ord, scen, wv_cur, week_range, z=None, Po=None, Pf=None): 

            self.TestType = TestType
            self.stage = stage
            self.x = x
            self.y = y
            self.week_range = week_range
            self.z = z
            self.Po = Po
            self.Pf = Pf
            self.wv = wv
            self.ord = ord
            self.scen = scen
            self.colors = [
                    "#AB1369",  # Dark Pink
                    "#E6A710",  # Yellow
                    "#008080",  # Teal
                    "#8FBCBA",  # Purple
                    "#A19DAF",  # Dark Green
                    "#FF0000",  # Red
                ]
            self.wv_cur = wv_cur

        def get_sessions(self, main, start_week):
            sesI = {}
            for theta in range(self.week_range[0], self.week_range[1]):
                sesI[start_week+theta] = {}
                for i in main.mod_dem:
                    for l in main.mod_locD[i]:
                        sesI[start_week+theta][i,l] = 0
                        for t in range(theta*main.num_sessions, theta*main.num_sessions+main.num_sessions): 
                            if self.x[i,t,l] >0.001:
                                sesI[start_week+theta][i,l]+=1
                            elif self.stage ==0: 
                                if self.z[i,t,l]>0.001:
                                    sesI[start_week+theta][i,l]+=1
            return sesI
        
       
        def results_op(self, main, wv_cur, start_week, ses_cur):
            res_gem_w = []
            res_max_w = []
            res_goal = []
            res_maxcap = []
            res_wv = []
            res_wv2 = []
            res_ses_dif = {}

            for theta in range(main.num_weeks):
                dev_goal = []
                dev_wv = []
                maxcap = 0
                wv_curvals = []
                wv_curvals2 = []
                for i in main.mod_dem:
                    prodI = 0
                    sesI = 0
                    sesC = 0
                    for ll, loc in enumerate(main.locs):
                        for l in set(loc)& set(main.mod_locD[i]):
                            sesC +=ses_cur[start_week+theta][i,l]
                            for t in range(theta*main.num_sessions, theta*main.num_sessions+main.num_sessions): 
                                if self.x[i,t,l] >0.001:
                                    sesI +=1
                                    # if i ==0:
                                    #     prodI += main.dict_prod_av[i]
                                    try:
                                        prodI += self.ord[i, theta+start_week, ll, 'p'] #use real production
                                    except KeyError:
                                        prodI += main.dict_prod_av[i]

                    #determine if aan max?
                    if sesI==main.dict_cap[i, 'max']:
                        maxcap+= 1
                    #determine changes in sessions, both percentage as deviation:
                    res_ses_dif[start_week+theta, i, "perc"] = (sesI-sesC)/sesC
                    res_ses_dif[start_week+theta, i, "ses"] = sesI-sesC #nieuw - oud, dus meer is positief, minder is negatief

                    #determine goal and wv KPIs
                    print("staples", main.dem_basis, main.num_sessions)
                    goal_ses = 4
                    if self.scen[0][i, theta, "wv"]/main.dict_prod_av[i] < main.dem_basis[i] - goal_ses:
                        goal = self.ord[i,theta+start_week, 'ro']/main.dict_prod_av[i]-goal_ses*0.25 -sesI
                        dev_goal.append(np.abs(goal)) #goal sessies van benodigde sessies
                    elif  main.dem_basis[i] - goal_ses <= self.scen[0][i, theta, "wv"]/main.dict_prod_av[i] < main.dem_basis[i] + goal_ses:
                        goal = self.ord[i,theta+start_week, 'ro']/main.dict_prod_av[i]-sesI
                        dev_goal.append(np.abs(goal)) #goal sessies van benodigde sessies
                    elif main.dem_basis[i] + goal_ses <= self.scen[0][i, theta, "wv"]/main.dict_prod_av[i] < main.dem_basis[i] + 2*goal_ses:
                        goal = self.ord[i,theta+start_week, 'ro']/main.dict_prod_av[i]+goal_ses*0.125- sesI
                        dev_goal.append(np.abs(goal))
                    elif self.scen[0][i, theta, "wv"]/main.dict_prod_av[i] >= main.dem_basis[i] + 2*goal_ses:
                        goal = self.ord[i,theta+start_week, 'ro']/main.dict_prod_av[i]+goal_ses*0.25- sesI
                        dev_goal.append(np.abs(goal))
                    # print("wv hiervoor", "week:",theta+start_week-1,  wv_cur[theta+start_week-1, i])
                    # print("voorspelling:", self.scen[0][i, theta, "wv"])
                    wv_cur[theta+start_week, i] = np.maximum(wv_cur[theta+start_week-1, i]+self.ord[i,theta+start_week, 'rw'] - prodI, 0) # in orders
                    wv_curvals.append(wv_cur[theta+start_week, i])
                    if main.S == 1:
                        wv_curvals2.append([self.scen[0][i, theta, "wv"], self.wv[i]])
                    else:
                        scens = []
                        for s in range(main.S):
                            scens.append(self.scen[s][i, theta, "wv"])
                        wv_curvals2.append([sum(scens)/main.S, self.wv[i]])
                        
                    dev_wv.append(np.abs(wv_cur[theta+start_week, i]/main.dict_prod_av[i] - main.dem_basis[i])/(main.dem_basis[i]/2))#dus weken deviation krijgen
                res_gem_w.append(sum(dev_wv)/len(main.mod_dem)) #in sessies
                res_max_w.append(np.max(dev_wv)) #in sessies
                res_goal.append(sum(dev_goal)/len(main.mod_dem)) #in sessies
                res_maxcap.append(maxcap)
                res_wv.append(wv_curvals)
                res_wv2.append(wv_curvals2)
            #determine fte 
            fte = []
            for theta in range(self.week_range[0], self.week_range[1]):
                ingezet_fte = 0
                for j in range(main.J):
                    for i in main.emp_skills[j]:
                        for t in main.emp_t[j,theta]: 
                            for l in set(main.mod_locD[i]) & set(main.emp_locD[j]):
                                if self.y[i, j, t, l] > 0.001:
                                    ingezet_fte += 1
                fte.append(ingezet_fte/9)
           # print(ingezet_fte, fte, 'FTE')

            return res_gem_w, res_max_w, res_goal, wv_cur, fte, res_maxcap, res_wv, res_wv2, res_ses_dif

        def fig_schedule_1st(self, main, file_name: str, scenario_num):
            #grootte figuur:
            figs=0
            for i, modality in enumerate(main.modalities):
                for l in main.mod_locD[i]:
                    figs+=1
            fig, ax = plt.subplots(figsize=(12, 0.4*figs))

            # Plot thick light grey horizontal bars for modalities and locations
            counter = 0
            ticks_gen = []
            for i, modality in enumerate(main.modalities):
                for l in main.mod_locD[i]:
                    ax.barh(counter, main.T, left=0, color='#FFFFFF', height=0.5, edgecolor='black', label=f'{modality} at Loc {l}')
                    counter +=1
                    loc_names = ["1", "2", "3"]
                    ll = next((loc_names[i] for i, sublist in enumerate(main.locs) if l in sublist), "NA")
                    ticks_gen.append(f'{modality} at Location {ll}')

            counter = 0
            
            for i in range(main.I):
                for l in main.mod_locD[i]:
                    num_reg=0
                    num_flex =0
                    for theta in range(self.week_range[0], self.week_range[1]):
                        for t in range(theta*main.num_sessions, theta*main.num_sessions+main.num_sessions): 
                            if self.stage ==0 or self.stage==2:
                                if self.x[i, t, l]> 0.001:
                                    num_reg+=1
                                    extra = 0
                                    theta = t// main.num_sessions
                                    tt= t - theta*main.num_sessions
                                    if (i,tt,l) in main.dict_bez_ext:
                                        extra = main.dict_bez_ext[i,tt,l]
                                    bez = main.dict_bez[i,l]+extra
                                    sub_bar_width = 0.5/bez 
                                    k = 0 #hoeveel verhoogt moeten worden
                                    for j in set(main.loc_emp[l]) & set(main.skills_empD[i])&set(main.t_empD[t]): 
                                        if self.y[i, j, t, l] > 0.001:
                                            if bez ==1:
                                                ax.barh(counter,1, left=t, color=self.colors[j], height=sub_bar_width, edgecolor='black',linewidth=0.6, alpha=1)
                                            elif bez==2:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1), 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==3:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/9, 1, left=t, color=self.colors[j], height=sub_bar_width,linewidth=0.6, edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==4:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/5.25, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==5:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/4, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                            elif self.stage ==1:
                                if self.x[i, t, l].x > 0.001:
                                    num_reg+=1
                                    extra = 0
                                    theta = t// main.num_sessions
                                    tt= t - theta*main.num_sessions
                                    if (i,tt,l) in main.dict_bez_ext:
                                        extra = main.dict_bez_ext[i,tt,l]
                                    bez = main.dict_bez[i,l]+extra
                                    sub_bar_width = 0.5/bez 
                                    k = 0 #hoeveel verhoogt moeten worden
                                    for j in set(main.loc_emp[l]) & set(main.skills_empD[i])&set(main.t_empD[t]): 
                                        if self.y[i, j, t, l].x > 0.001:
                                            if bez ==1:
                                                ax.barh(counter,1, left=t, color=self.colors[j], height=sub_bar_width, edgecolor='black',linewidth=0.6, alpha=1)
                                            elif bez==2:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1), 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==3:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/9, 1, left=t, color=self.colors[j], height=sub_bar_width,linewidth=0.6, edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==4:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/5.25, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                                            elif bez==5:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/4, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1)
                                                k+=1
                            if self.stage==0:
                                if self.z[i, t, l] > 0.001:
                                    num_flex+=1
                                    extra = 0
                                    if (i,t,l) in main.dict_bez_ext:
                                        extra = main.dict_bez_ext[i,t,l]
                                    bez = main.dict_bez[i,l]+extra
                                    sub_bar_width = 0.5/bez 
                                    k = 0 #hoeveel verhoogt moeten worden
                                    for j in set(main.loc_emp[l]) & set(main.skills_empD[i])&set(main.t_empD[t]):
                                        if self.y[i, j, t, l] > 0.001:
                                            if bez ==1:
                                                ax.barh(counter,1, left=t, color=self.colors[j], height=sub_bar_width, edgecolor='black', alpha=1, hatch='/')
                                            elif bez==2:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1), 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1, hatch='/')
                                                k+=1
                                            elif bez==3:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/9, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1, hatch='/')
                                                k+=1
                                            elif bez==4:
                                                ax.barh(counter + k*sub_bar_width - sub_bar_width/bez*(bez-1)-0.5/5.25, 1, left=t, color=self.colors[j], height=sub_bar_width, linewidth=0.6,edgecolor='black', alpha=1, hatch='/')
                                                k+=1
                            else: 
                                num_flex=0
                    ax.text(main.T+0.2, counter, f"{num_reg} : {num_flex}", va='center', ha='left', fontsize=8)
                    counter+=1
                    ax.text(main.T+0.1, 7, "#reg:#flex", va='center', ha='left', fontsize=8) #counter =9
            legend_handles = []
            for j in range(main.J):
                skills_indices = main.emp_skills[j]
                skills_names = [main.modalities[s] for s in skills_indices]
                legend_label = f'{j+1}: FTE={16}, Skills={", ".join(skills_names)}'
                legend_handles.append(mpatches.Patch(color=self.colors[j], label=legend_label))
            
            legend_handles.append(mpatches.Patch(facecolor="#FFFFFF", edgecolor="black", label='Regular session'))
            legend_handles.append(mpatches.Patch(facecolor="#FFFFFF", edgecolor="black", hatch='//', label='Flexible session'))

            ax.legend(handles=legend_handles, title='Employees', loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=3)

            
            # Customize the plot
            ax.set_yticks(range(counter))
            ax.set_yticklabels(ticks_gen)
            # Set x-ticks for both subplots
            xticks = np.arange(0, main.num_sessions*main.num_weeks, main.num_sessions)
            ax.set_xticks(xticks)
            xticklabels = [str(i) for i in range(main.num_weeks)]
            ax.set_xticklabels(xticklabels)

            ax.set_xlabel('Week (sessions)')
            ax.set_title(f'Tactical Decision PI: Sessions and Employees Schedule')
            if self.stage==1:
                ax.set_title(f'Scenario {scenario_num+1}')
            # Add the textbox below the legend
            #plt.text(0.92, 0.8, f"n", transform=plt.gcf().transFigure, fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
            if self.stage==0 or self.stage==2:
                for i in main.mod_dem:
                    extra_text = f"{main.modalities[i]}: W: {self.wv[i]}, D: {sum(self.ord[i,theta] for theta in range(main.num_weeks))}, per θ: {self.ord[i,0]}, {self.ord[i,1]}, {self.ord[i,2]}, {self.ord[i,3]}"
                    plt.text(0.94, 0.2 + 0.15 * i, extra_text, transform=plt.gcf().transFigure, fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
            if self.stage==1:
                for i in main.mod_dem:
                    extra_text = f"{main.modalities[i]}: W: {self.scen[i,0,'wv']}, D: {sum(self.scen[i,theta, 'o'] for theta in range(main.num_weeks))}, per θ: {self.scen[i,0, 'o']}, {self.scen[i,1, 'o']}, {self.scen[i,2, 'o']}, {self.scen[i, 3, 'o']}"
                    plt.text(0.94, 0.2 + 0.15 * i, extra_text, transform=plt.gcf().transFigure, fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))


            plt.savefig(file_name, bbox_inches='tight')
            plt.close()


   