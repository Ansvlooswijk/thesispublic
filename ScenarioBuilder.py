def scenario_builder(S, agg, num_weeks):
    scenariotree= []
    for s in range(S):
        if not agg: 
            for theta in range(num_weeks):
                week_range = [theta, theta+1]
                scenariotree.append([s, week_range])
        else: 
            scenariotree.append([s, [0, num_weeks]])
    return scenariotree