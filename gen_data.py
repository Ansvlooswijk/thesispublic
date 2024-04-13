import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate DataFrame

def gen_data(genseed):

    ctel = 2.5 #+ np.random.normal(0,0.25)
    ctem = 20 #+ np.random.normal(0,2)

    mriel = 7#+ np.random.normal(0,0.7)
    mriem = 5#+ np.random.normal(0,0.5)

    weeks = list(range(68))
    planned_as_values = [10 if week not in [15, 21, 32, 33, *range(42, 48), 54, 66] else 6 for week in range(68)]
    realized_as_valuesCT = [int(np.random.normal(planned_as, 1)) for planned_as in planned_as_values]
    realized_as_valuesMRI = [int(np.random.normal(planned_as, 1)) for planned_as in planned_as_values]
    wv_valuesMRI = [36] * 68
    wv_valuesCT= [18] * 68
    prod_ct_values = [int(np.random.normal(15, 0.5)) for _ in range(68)]
    prod_mri_values = [int(np.random.normal(15, 0.5)) for _ in range(68)]
    ct_el_values = [int(round(sum(np.random.normal(ctel, 0.5, ra)))) for ra in realized_as_valuesCT]
    ct_em_values = [int(np.random.normal(ctem, 2)) for _ in range(68)]
    mri_el_values = [int(round(sum(np.random.normal(mriel, 0.5, ra)))) for ra in realized_as_valuesMRI]
    mri_em_values = [max(0, round(np.random.normal(mriem, 2))) for _ in range(68)]

    data = {
        'Week': weeks,
        'Planned_AS': planned_as_values,
        'CTRealized_AS': realized_as_valuesCT,
        'MRIRealized_AS': realized_as_valuesMRI,
        'WVMRI': wv_valuesMRI,
        'WVCT': wv_valuesCT,
        'ProdCT': prod_ct_values,
        'ProdMRI': prod_mri_values,
        'CTel': ct_el_values,
        'CTem': ct_em_values,
        'MRIel': mri_el_values,
        'MRIem': mri_em_values
    }

    df = pd.DataFrame(data)

    #introduce 'random' noise: 
    # for week in range(len(df)):
    #     if random.random() > 0.75:
    #         if random.random() >0.5:
    #             df.CTem[week] = df.CTem[week] + round(np.random.normal(0,5))
    #         if random.random() >0.5:
    #             df.CTem[week] = df.MRIem[week] + round(np.random.normal(0,5))

    #introduce deviations for a prolonged (4-weeks) period of times
    # ind = True
    # for w in range(0,68,4):
    #     for week in range(w,w+4):
    #         if ind: 
    #             df["CTRealized_AS"][week] = 6
    #             df["MRIRealized_AS"][week] = 10
    #         else: 
    #             df["CTRealized_AS"][week] = 10
    #             df["MRIRealized_AS"][week] = 6
    #     if ind: 
    #         ind = False
    #     else: 
    #         ind = True

    df.to_csv('generated_data.csv', index=False)
