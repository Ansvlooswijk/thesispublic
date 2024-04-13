import pandas as pd
import ast
import numpy as np
import pickle
import logging
from scipy.stats import norm

from InputfileHandler import input_file_handler 
import logging

class InputCreator():
    def __init__(self, test_type, start_week, num_weeks, current_week):
        (self.csv_mod, self.csv_emp, self.csv_av, self.csv_scen, self.csv_prod, self.csv_ks, _, _, _, self.locs, _, 
         self.dem_basisO, self.num_sessions, self.FTE_correctie, self.closed_days, self.dict_prod_av, self.dict_prod) = input_file_handler(test_type, current_week)
        self.start_week = start_week
        self.num_weeks = num_weeks

    def General(self):
        mod = pd.read_csv(self.csv_mod, encoding='utf-8-sig', sep=";" )
        emp = pd.read_csv(self.csv_emp, encoding='utf-8-sig', sep=";" )
        av = pd.read_csv(self.csv_av)
        prod = pd.read_csv(self.csv_prod)
        ks = pd.read_csv(self.csv_ks) 

        T = self.num_sessions*self.num_weeks
        
        modalities = mod.Modality.unique().tolist()
        print(modalities)
        I = len(modalities)

        mod['mod_index'] = mod['Modality'].apply(lambda mod: modalities.index(mod) if mod in modalities else -1)

        L=len(mod.Location.unique().tolist())
        loc_mod = {} #per location which modalities are there
        for l in range(L):
            rows_with_location = []
            rows_with_location = mod[mod['Location'] == l]
            loc_mod[l]= rows_with_location['mod_index'].unique()

        mod_locD = {} #per modality which locations are there
        for i in range(I):
            rows_with_location = []
            rows_with_location = mod[mod['mod_index'] == i]
            mod_locD[i]= rows_with_location['Location'].unique()

        #write min [0] and max [1] dictionaries: 
        dict_cap = {}
        dict_cap_month = {}
        for i in range(I): 
            tot =0
            for l in mod_locD[i]:
                result = mod.loc[(mod['Location'] == l) & (mod['mod_index'] == i)]
                if not result.empty:
                    dict_cap[i,l] = result[['Min', 'Max']].iloc[0].astype(int).tolist()
                    tot+= result['Max'].iloc[0].astype(int)
                    monthmin= result[['MonthMin']].iloc[0]
                    if monthmin.any()>0:
                        dict_cap_month[i] = monthmin.iloc[0]
                else: 
                    logging.warning("in dict_cap not all locations have a min/max for modality")
            dict_cap[i, 'max'] = tot
        dict_bez = {}
        for l in range(L):
            for i in loc_mod[l]: 
                result = mod.loc[(mod['Location'] == l) & (mod['mod_index'] == i)]
                if not result.empty:
                    dict_bez[i, l] = result['Bezetting'].iloc[0].astype(int)
                else: 
                    logging.warning("WARNING: in dict_bez not all locations have a bezetting for modality")

        dict_verpl={}
        for row in range(len(mod)):
            if pd.notna(mod.at[row, 'Verplicht']):
                current_list = ast.literal_eval(mod['Verplicht'][row])
                dict_verpl[mod['mod_index'][row], int(mod['Location'][row])]= current_list

        dict_bez_ext={}
        for row in range(len(mod)):
            if pd.notna(mod.at[row, 'Bezetting_extra']):
                current_list = ast.literal_eval(mod['Bezetting_extra'][row])
                for t in current_list:
                    dict_bez_ext[mod['mod_index'][row],t,int(mod['Location'][row])]= int(mod['Aantal'][row])        

        #needed vectors employee related
        J = len(emp)
        emp['Echo'] = np.where((emp['Echo_lab'] == 1) | (emp['Echo_ass'] == 1), 1, 0)

        skills_empD = {} #all employees with a certain modality skill
        general_skills = ["Bucky" ,"Block" ,"Flex", "Protocol"]
        for index, modality in enumerate(modalities):
            if modality in general_skills:
                non_zero_indices = emp.index[emp["General"] != 0].tolist()
            else:
                non_zero_indices = emp.index[emp[modality] != 0].tolist()
            skills_empD[index] = non_zero_indices
        #skills_dict[index]
        emp_skills = {} #all skills of a certain employee
        for row in range(len(emp)):
            non_zero_columns = []
            for i, modality in enumerate(modalities):
                if modality in general_skills:
                    if emp["General"][row] != 0:
                        non_zero_columns.append(i)
                else:
                    if emp[modality][row] != 0:
                        non_zero_columns.append(i)
            emp_skills[emp.ID[row]] = non_zero_columns

        #Keepskill
        genplus = general_skills + ["Interne"]
        index_general = [index for index, value in enumerate(modalities) if value in genplus]
        ks['i'] = ks['Modality'].apply(lambda mod: [modalities.index(mod)] if mod in modalities else index_general)
        dict_k=ks.set_index('Modality').to_dict(orient='index')          

        #Location Panningen
        loc_emp = {} #per location which employees
        for l in range(L):
            if l==L-1:
                rows_with_location = emp[emp['Panningen'] == 1]
                loc_emp[L-1]= rows_with_location['ID'].unique()
            else:
                loc_emp[l]= range(J)

        emp_locD = {} # per employee which locations
        for j in range(J):
            if emp['Panningen'][j]==1:
                emp_locD[j] =  range(L)
            else:
                emp_locD[j] =  range(L-1)
            #FTE aftrekken voor CT diensten en algemene diensten (die rooster ik dus niet in, ik reserveer er slechts fte voor. De FTE worden gelijkmatig verdeeld.)

        if self.FTE_correctie: 
            CT_skilled = emp.loc[emp['CT'] == 1, 'FTE'].sum()
            Dienst_skilled = emp.loc[emp['CT'] != 1, 'FTE'].sum()
            full_fte = 36
            CT_dienst = 16*8/(CT_skilled*full_fte)  #16 diensten per week / hoeveel CT ers is dus hoeveel uur per week er van iemand afgaat voor de shifts
            dienst = 9*8/(Dienst_skilled*full_fte)
            emp.loc[emp['CT'] == 1, 'FTE'] -= CT_dienst
            emp.loc[emp['CT'] != 1, 'FTE'] -= dienst
            emp.loc[emp['FTE'] < 0, 'FTE'] = 0

        FTE = emp.FTE.tolist() 
        sumft = 0
        for j in range(len(FTE)):
            FTE[j]=round(FTE[j]*self.num_weeks*9) #fte afronden naar dichtstbijzijnde hele getal
            sumft += FTE[j]

        emp_t  ={} #per employee which times available
        t_empD = {} #per time which employees available 
        for t in range(self.num_sessions*self.num_weeks):
            t_empD[t]=[]
        for j in range(J):
            for index, theta in enumerate(range(self.start_week, self.start_week+self.num_weeks)):
                result = av.loc[(av['ID'] == j) & (av['Week'] == theta)]
                week = range(index*self.num_sessions, index*self.num_sessions + self.num_sessions)
                if not result.empty:
                    afw = result['Roostervrij'].iloc[0]
                    verl = result['Verlof'].iloc[0]
                    current_list0 = []#ast.literal_eval(afw)
                    current_list1 = ast.literal_eval(verl) 
                    current_list = current_list0 + current_list1
                    current_listn = [element + index*self.num_sessions for element in  current_list]
                    missing_elements = [t for t in week if t not in current_listn]
                    emp_t[j,index]= missing_elements
                    for t in missing_elements:
                        t_empD[t].append(j)
                else:
                    emp_t[j,index]= week
                    for t in week:
                        t_empD[t].append(j)

        roost_emp_t  ={} #per employee which times eigenlijk roostervrij 

        for j in range(J):
            for index, theta in enumerate(range(self.start_week, self.start_week+self.num_weeks)):
                result = av.loc[(av['ID'] == j) & (av['Week'] == theta)]
                week = range(index*self.num_sessions, index*self.num_sessions + self.num_sessions)
                if not result.empty:
                    afw = result['Roostervrij'].iloc[0]
                    current_list0 = ast.literal_eval(afw)
                    current_list = [element + index*self.num_sessions for element in current_list0]
                    roost_emp_t[j,index]= current_list
                else:
                    roost_emp_t[j,index]= []

        # prod['mod_index'] = prod['Type'].apply(lambda d: modalities.index(d) if d in modalities else -1) 
        # dict_prod = {}
        # print(prod)
        # for index,loc in enumerate(self.locs): 
        #     for l in loc: 
        #         for i in set(loc_mod[l]):
        #             result = prod.loc[(prod['mod_index'] == i) & (prod['Location'] ==index)]
        #             if not result.empty:
        #                 dict_prod[i,l]= result['Productivity'].iloc[0]
        #             else:
        #                 logging.warning(f"Warning not all locations have production: {i}, l {l}, loc {index}")

        # import pickle
        # with open('dict_prod_ext.pickle', 'wb') as file:
        #     pickle.dump(dict_prod, file)       

        #Get average duration of order: sessie/order
        #duur = prod.groupby(['mod_index','Type']).mean().reset_index()
        #dict_prod_av = np.zeros(len(modalities))
        # Iterate over the rows of the DataFrame
        # for i in range(len(duur)): 
        #     dict_prod_av[duur.mod_index[i]] = duur['Productivity'][i]
        
        # import pickle
        # with open('dict_prod2_av.pickle', 'wb') as file:
        #     pickle.dump(dict_prod_av, file)
        # print(dict_prod_av)
        # print(dict_prod)

        dict_costs= {}
        for row in range(len(mod)):
            dict_costs[mod.mod_index[row], mod.Location[row]] = mod.Costs[row]


        return L, J, I, loc_mod, mod_locD, dict_bez, dict_cap,dict_cap_month, dict_verpl,dict_bez_ext, skills_empD, emp_skills, \
                loc_emp, emp_locD, emp_t, t_empD, roost_emp_t, self.dict_prod_av, self.dict_prod, dict_costs, dict_k, FTE, self.closed_days,\
            modalities, self.locs, self.num_sessions, T
 
def get_demand_gen(dem_inputs, S, seedsettings):
    type_dem, type_wv, horizon, current_week, num_weeks, start_week, ses_cur, wv_cur, _, _ = dem_inputs
    np.random.seed(seedsettings)
    scen = pd.read_csv("generated_data.csv")

    dem_modalitiesI = [0,1]
    modalities = ["CT", "MRI"]

    dict_ord0 = {} #contains a list: urg, ele
    dict_wv0 = {}
    dict_scen = {} #contains a list: urg, ele

    prod_av =15
    meani = [2.5, 7]
    meanemi = [20, 5]
    devemi = [2,2]

    for s in range(S):
        dict_scen[s] = {}
        dict_scen[s]['p']=1/S
        #wv

    for i in dem_modalitiesI:
        #get planned sessions = outflow
        productie=0
        for theta in range(current_week, start_week):
            dict_ses = ses_cur[theta]
            for (ii,l), val in dict_ses.items():
                if ii==i:
                    productie += prod_av*val #temporary 

        #make order prediction
        ord_gen0=0
        ord_gen1=0

        for w2 in range(current_week-2, current_week):#first two weeks top 10
            ord_gen0 += scen.loc[(scen['Week'] == w2), f'{modalities[i]}el'].values[0]
            ord_gen1 += scen.loc[(scen['Week'] == w2), f'{modalities[i]}el'].values[0]

        for w in range(current_week, current_week+2): #top 10 departments for each week week 3, 4 zijn sessies bekend
            sessions_AS = scen.loc[(scen['Week'] == w), f'{modalities[i]}Realized_AS'].values[0]
            ord_gen0  += meani[i]*(sessions_AS) #use mean
                    
            #generate orders1
            if type_dem == "M":                       
                frac_ord = np.random.normal(meani[i], 1, size=S)
                ord_gen1 += frac_ord*(sessions_AS)
            elif type_dem == "25th":
                ord_gen1  += norm.ppf(0.25, loc=meani[i], scale=0.5)*(sessions_AS)
            elif type_dem == "mean":
                ord_gen1  += meani[i]*(sessions_AS)
            elif type_dem == "75th":
                ord_gen1 += norm.ppf(0.75, loc=meani[i], scale=0.5)*(sessions_AS)
            else: 
                print("Define type_dem")
                break

        for w in range(current_week+2, start_week-2): #top 10 departments for each week sessies nog onbekend
            sessions_AS = scen.loc[(scen['Week'] == w), f'Planned_AS'].values[0]
            ord_gen0  += meani[i]*(sessions_AS) #use mean

            if type_dem == "M":                       
                sessions_AS_dev = int(np.random.normal(0, 1))
                frac_ord = np.random.normal(meani[i], 1, size=S)
                ord_gen1 += frac_ord*(sessions_AS + sessions_AS_dev)
            elif type_dem == "25th":
                ord_gen1  += norm.ppf(0.25, loc=meani[i], scale=0.5)*(sessions_AS + norm.ppf(0.25, loc=0, scale=1))
            elif type_dem == "mean":
                ord_gen1  += meani[i]*(sessions_AS)
            elif type_dem == "75th":
                ord_gen1 += norm.ppf(0.75, loc=meani[i], scale=0.5)*(sessions_AS + norm.ppf(0.75, loc=0, scale=1))
            else: 
                print("Define type_dem")
                break                
        #em
        for h in range(horizon): 
            ord_gen0 += meanemi[i] #for 12 weeks

            if type_dem == "M":                       
                ord_gen1 += np.random.normal(meanemi[i], devemi[i], size=S)
            elif type_dem == "25th":
                ord_gen1  += norm.ppf(0.25, loc=meanemi[i], scale=devemi[i])
            elif type_dem == "mean":
                ord_gen1  += meanemi[i]
            elif type_dem == "75th":
                ord_gen1 += norm.ppf(0.75, loc=meanemi[i], scale=devemi[i])
            else: 
                print("Define type_dem")
                break   
   
        dict_wv0[i] = round((wv_cur[current_week, i] + ord_gen0 - productie)/prod_av) #van orders naar sessies
        for s in range(S):
            for theta in range(num_weeks):
                if S==1:
                    wv_samp = wv_cur[current_week, i]+ ord_gen1-productie #in orders
                else: 
                    wv_samp = wv_cur[current_week, i]+ ord_gen1[s]-productie #in orders
                dict_scen[s][i, theta, 'wv'] = round(wv_samp/prod_av) #in sessies

    for i in dem_modalitiesI:
        for theta in range(num_weeks):
            sumAS0 = 0
            sumAS1 = np.zeros(S)
            ord_gen = []
            if horizon==12:
                sessions_AS = scen.loc[(scen['Week'] == theta+start_week-2), f'Planned_AS'].values[0]
                sumAS0 += meani[i]*(sessions_AS) #val2=mean
                #generate orders1
                if type_dem == "M":                       
                    for s in range(S):
                        ords = 0
                        sessions_AS_dev = int(np.random.normal(0, 1))
                        for ses in range(sessions_AS+sessions_AS_dev):
                            ords += np.random.normal(meani[i],1)
                        ord_gen.append(ords)
                elif type_dem == "25th":
                    ord_gen  = norm.ppf(0.25, loc=meani[i], scale=0.5)*(sessions_AS + norm.ppf(0.25, loc=0, scale=1))
                elif type_dem == "mean":
                    ord_gen  = meani[i]*(sessions_AS)
                elif type_dem == "75th":
                    ord_gen = norm.ppf(0.75, loc=meani[i], scale=0.5)*(sessions_AS + norm.ppf(0.75, loc=0, scale=1))
                else: 
                    print("Define type_dem")
                    break
                sumAS1 += ord_gen
            elif horizon == 4:
                sessions_AS = scen.loc[(scen['Week'] == theta+start_week-2), f'{modalities[i]}Realized_AS'].values[0]
                sumAS0 += meani[i]*(sessions_AS) #val2=mean
                #generate orders1
                if type_dem == "M":  
                    for s in range(S):
                        ords =0
                        sessions_AS_dev = int(np.random.normal(0, 1))
                        for ses in range(sessions_AS+sessions_AS_dev):
                            ords += np.random.normal(meani[i], 1)
                        ord_gen.append(ords)
                elif type_dem == "25th":
                    ord_gen  = norm.ppf(0.25, loc=meani[i], scale=0.5)*(sessions_AS)
                elif type_dem == "mean":
                    ord_gen  = meani[i]*(sessions_AS)
                elif type_dem == "75th":
                    ord_gen = norm.ppf(0.75, loc=meani[i], scale=0.5)*(sessions_AS)
                else: 
                    print("Define type_dem")
                    break
                sumAS1 += ord_gen
            sumAS0 += meanemi[i] 
            ord_gen =0
            if type_dem == "M":                        
                ord_gen = np.random.normal(meanemi[i], devemi[i], size=S)
            elif type_dem == "25th":
                ord_gen  = norm.ppf(0.25, loc=meanemi[i], scale=devemi[i])
            elif type_dem == "mean":
                ord_gen  = meanemi[i]
            elif type_dem == "75th":
                ord_gen = norm.ppf(0.75, loc=meanemi[i], scale=devemi[i])
            else: 
                print("Define type_dem")
                break
            #generate orders1
            sumAS1 += ord_gen              

            dict_ord0[i, theta] = np.round(sumAS0/prod_av)
            for s in range(S):
                dict_scen[s][i, theta, 'o'] = np.round(sumAS1[s]/prod_av) 
    
    #Get true orders and production:
    for i in dem_modalitiesI:
        for theta in range(num_weeks):
            ordRO= scen.loc[(scen['Week'] == theta+start_week), f'{modalities[i]}em'].values[0]
            ordRO+= scen.loc[(scen['Week'] == theta+start_week), f'{modalities[i]}el'].values[0] 
            dict_ord0[i, theta+start_week, 'rw'] = ordRO

            ordRW= scen.loc[(scen['Week'] == theta+start_week), f'{modalities[i]}em'].values[0]
            ordRW+= scen.loc[(scen['Week'] == theta+start_week-2), f'{modalities[i]}el'].values[0] 
            dict_ord0[i, theta+start_week, 'ro'] = ordRW

            for l in range(5):
                prodR= scen.loc[(scen['Week'] == theta+start_week), f'Prod{modalities[i]}'].values[0]
                dict_ord0[i, theta+start_week, l, 'p'] =prodR

    newS = S
    dict_wvster = [12,12]

    return dem_modalitiesI, dict_wv0, dict_ord0, dict_scen, newS, dict_wvster
    


        


            

