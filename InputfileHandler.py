    
import pickle

def input_file_handler(test_type, current_week): 
    if test_type==2:
        csv_mod = 'test_data/input_mod_simp.csv' 
        csv_emp  = 'test_data/input_emp_simp.csv'  
        csv_av = 'test_data/OfTEST_besch.csv'
        csv_scen = None
        csv_prod = "test_data/OfTEST_prod.csv"
        csv_ks = "test_data/input_ks_simp.csv"
        csv_sas =  None
        csv_smbv = None
        csv_wv = None
        locs = [[0,1,2,3,4],[5]]
        mod_dem= ['CT', 'MRI']
        dembasis = [8, 8]
        num_sessions = 4
        FTE_correctie = False
        closed_days = []
        # with open('test_data/dict_prodTEST_av.pickle', 'rb') as file:
        #     dict_prod_av =  pickle.load(file)
        with open('test_data/dict_prodTEST.pickle', 'rb') as file:
            dict_prod =  pickle.load(file)
        dict_prod_av = {}
        dict_prod_av[0] = 15
        dict_prod_av[1] = 15
        for key, value in dict_prod.items():
            dict_prod[key] = value*1.5

    return csv_mod, csv_emp, csv_av, csv_scen, csv_prod, csv_ks, csv_sas, csv_smbv, csv_wv, locs, mod_dem, dembasis, num_sessions, FTE_correctie, closed_days, dict_prod_av, dict_prod