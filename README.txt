The code in this GitHub was developed for my masters thesis called: A flexible scheduling framework for a radiology department: Integrating workforce scheduling and 
resource allocation with a multi-stage stochastic program on a rolling horizon based on supply and demand

In short, this code was written to evaluate a two-stage stochastic program using S scenarios to schedule the facilities and employees of a radiology department.
Gurobi was used as solver and a callback function was written to implement the L-shaped method. Testdata is generated and subsequently, the scheduling decisions are 
taken over a horizon of 12 months. 

A brief description of the files within this GitHub: 
- folder test_data contains test data sets for testing purposes. 
- gen_data generates a data csv that is used to evaluate the performance of the model over a year
- InputCreator generates a collection of functions, dictionaries and other inputs necessary for the model
- InputFileHandler handles all the input files. The inputfiles contain the characteristics of the radiology department
- main.py is the main function that should be run to start the code. It contains multiple custumization options. 
- MasterBuilder builds the master problem
- ModelBuilder contains all constraints, objectives and evaluation functions relating to the mixed integer linear program
- MyCallback implements the integer L-shaped method to find solutions to the two-stage stochastic program
- ScenarioBuilder build the different scenarios
- SubBuilder build the subproblems

Notes: 
- The code should be self sufficient with the provided test-set. The results are saved into .pkl files. The progress is written 
to a logging file.
- The code is scarcely annotated with a mixture of Dutch and English comments, sorry :/. 

