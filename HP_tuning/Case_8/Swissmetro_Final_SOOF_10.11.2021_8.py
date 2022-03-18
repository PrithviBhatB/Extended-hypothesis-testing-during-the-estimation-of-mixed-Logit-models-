#!/usr/bin/env python
# coding: utf-8

# ## Optimization-based framework to assist analysts in generating and testing meaningful DCM specifications

# ## 1. Installation of discrete choice estimation package developed by QUT

# In[1]:


'pip install searchlogit -U'


# # Install a pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install xlogitprit -U

# # I. Initial steps Before the search

# ## 2. Importing libraries required for the algorithm

# In[2]:


import numpy as np 
import pandas as pd
import time 
import datetime # To report the total estimation time
import matplotlib.pyplot as plt # To plot graph showing improvement in BIC over iterations
from searchlogit import MixedLogit #for estimation of MNL models
from searchlogit import MultinomialLogit #for estimation of mixed logit models
import sys #to create outfile file
import math


# ## 3. Creating an output to store all estimations from the search process

# In[3]:


current_date = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
code_name = 'Swissmetro_August_' #Name for the search 
bench_mark =  code_name + 'benchmark_output_' + current_date + '.txt'
sys.stdout = open(bench_mark,'wt')
print("Search initiated at",time.ctime()) #print start time
sys.stdout.flush()


# ## 4. Importing dataset

# ### The present version of xlogitprit requires the dataset to be provided in csv form

# ### Swissmetro dataset

# In[4]:


#file path & file name
df = pd.read_csv("Swissmetro_final.csv")
df.columns


# ## 5.a. Inputs for Benchmark model

# ### If the modeller needs to compare a benchmark multinomial logit model with the results from the search, then it can be estimated here. 
# ### Following  inputs need to be provided by the modeller, which is required for the model estimation.

# In[5]:


choice_id = df['custom_id']
ind_id =df['ID']
base_varnames = ['TIME', 'COST', 'HEADWAY', 'SEATS','GA','AGE','LUGGAGE']
base_asvarnames = ['TIME', 'COST', 'HEADWAY', 'SEATS'] #['TIME', 'COST', 'HEADWAY','SEATS']
base_isvarnames = ['GA','AGE','LUGGAGE']
choice_set=['TRAIN','CAR','SM']
choice_var = df['CHOICE']
alt_var = df['alt']
Tol = 1e-2
base_intercept = True
#dist = ['n', 'ln', 'tn', 'u', 't', 'f']
av = df['AV']
weight_var = None
base = 'SM'
#dist = ['n', 'tn', 'u', 'f']
#"""


# ## 5.b. Estimation of Benchmark model

# In[6]:


from searchlogit import MultinomialLogit
model = MultinomialLogit()
model.fit(X=df[base_varnames].values, y=choice_var, varnames=base_varnames,isvars=base_isvarnames, alts=alt_var,
          ids=choice_id, avail=av,fit_intercept=True, base_alt = base,gtol=1e-01)
model.summary()
base_model = [1000000,[],[],{},[],[],False]
#print(base_model)


# ## 5.c. Variable Transformations

# ### function not used presently

# ### When lognormal distribution is used for random parameters and we expect the mean coefficient's sign to be positive, we convert the dataframe variables corresponding to those particular coefficients, to their opposite signs.
# ### For example, in the electricity dataset, we expect negative coefficients for pf, cl, tod and seas

# In[7]:


#list of variables that need transformation when the model is estimated with no intercept
ln_trans = [var for var in base_varnames if model.coeff_[base_varnames.index(var)] < 0]

#when model has intercept
#ln_trans = [var for var in base_asvarnames if model.coeff_[base_asvarnames.index(var)+(len(choice_set)-1)] < 0]
ln_trans =[]


# ## 5.e. Inputs for search

# In[8]:


boxc_l = ['L1','L2'] #['L1','L2','L3'] ##boxc_l is the list of suffixes used to denote manually transformed variables
l1 = np.log
l2 =  0.5
#l3 = 2

## Time and Cost variables have been transformed and included in the dataset
"""
#for PRICE
df['TIME_L1'] = l1(df['TIME'] + 1)
df['TIME_L2'] = (((df['TIME']**l2)-1)/l2)


#for COST
df['COST_L1'] = l1(df['COST'] + 1)
df['COST_L2'] = (((df['COST']**l2)-1)/l2)



#for HEADWAY
df['HEADWAY_L1'] = l1(df['HEADWAY'] + 1)
df['HEADWAY_L2'] = (((df['HEADWAY']**l2)-1)/l2)


"""
#for PRICE
df['TIME_L1'] = l1(df['TIME'] + (df['TIME'] == 0))
df['TIME_L1'] = df['TIME_L1'].apply(lambda x : x if x > 0 else 0)

df['TIME_L2'] = ((df['TIME']**l2)-1)/l2
df['TIME_L2'] = df['TIME_L2'].apply(lambda x : x if x > 0 else 0)

#for COST
df['COST_L1'] = l1(df['COST'] + (df['COST'] == 0))
df['COST_L1'] = df['COST_L1'].apply(lambda x : x if x > 0 else 0)

df['COST_L2'] = ((df['COST']**l2)-1)/l2
df['COST_L2'] = df['COST_L2'].apply(lambda x : x if x > 0 else 0)


#for HEADWAY
df['HEADWAY_L1'] = l1(df['HEADWAY'] + (df['HEADWAY'] == 0))
df['HEADWAY_L1'] = df['HEADWAY_L1'].apply(lambda x : x if x > 0 else 0)

df['HEADWAY_L2'] = ((df['HEADWAY']**l2)-1)/l2
df['HEADWAY_L2'] = df['HEADWAY_L2'].apply(lambda x : x if x > 0 else 0)


# In[9]:


#Modeller needs to provide a list of asvars that were manually transformed
trans_asvars = ['TIME','COST', 'HEADWAY'] #['PRICE','RECRE']


# In[10]:


choice_id = df['custom_id']
ind_id =df['ID']

varnames = ['TIME_L1','COST_L2', 'HEADWAY_L1','LUGGAGE','MALE','GA'] #all explanatory variables to be included in the model
"""
varnames = ['TIME', 'COST', 'HEADWAY','TIME_L1', 'TIME_L2', 'COST_L1', 'COST_L2', 'HEADWAY_L1', 'HEADWAY_L2',
       'SEATS', 'AGE_TRAIN', 'AGE_CAR', 'AGE_SM', 'LUGGAGE_CAR', 'LUGGAGE_SM',
       'LUGGAGE_TRAIN', 'INCOME_CAR', 'INCOME_SM', 'INCOME_TRAIN', 'MALE_CAR',
       'MALE_SM', 'MALE_TRAIN', 'WHO_CAR', 'WHO_SM', 'WHO_TRAIN',
       'FIRST_TRAIN', 'FIRST_SM', 'FIRST_CAR']
"""
asvarnames = ['TIME_L1','COST_L2', 'HEADWAY_L1'] # alternative-specific variables in varnames
isvarnames = ['LUGGAGE','MALE','GA'] # individual-specific variables in varnames
rvars = {'TIME_L1':'u','COST_L2':'t', 'HEADWAY_L1':'t'}
choice_set=['TRAIN','CAR','SM'] #list of alternatives in the choice set as string
choice_var = df['CHOICE'] # the df column name containing the choice variable
alt_var = df['alt'] # the df column name containing the alternative variable
av = None #df['AV']  #the df column name containing the alternatives' availability
weight_var = None #the df column name containing the weights 
base = 'SM' #reference alternative
R = 200 # number of random draws for estimating mixed logit models
Tol = 1e-4 #Tolerance value for the optimazition routine used in maximum likelihood estimation (default value is 1e-06)
iterations = 200 #number of iterations for the MLE optimization routine (default value is 2,000)
#dist = ['n', 'f'] 
dist = ['n', 'u', 't', 'ln', 'f'] #List of random distributions to select from 
#('n': normal; 'ln': lognormal;'tn': truncated-normal; 'u': uniform; 't': triangular; 'f': fixed)


# In[11]:


from searchlogit import MultinomialLogit
model = MultinomialLogit()
model.fit(X=df[varnames].values, y=choice_var, varnames=varnames,isvars=isvarnames, alts=alt_var,
          ids=choice_id, avail=av,fit_intercept=True, base_alt = base,gtol=1e-01)
model.summary()
base_model = [1000000,[],[],{},[],[],False]
#print(base_model)


# In[12]:


from searchlogit import MixedLogit
model = MixedLogit()
model.fit(X=df[varnames].values, y=choice_var, varnames=varnames,isvars=isvarnames, randvars=rvars,
          alts=alt_var,
          ids=choice_id,panels=ind_id, avail=av,fit_intercept=True, base_alt = base,gtol=1e-01)
model.summary()
base_model = [1000000,[],[],{},[],[],False]
#print(base_model)


# ### converting all negative values to zero
# df[df < 0] = 0

# In[13]:


choice_id = df['custom_id']
ind_id =df['ID']

varnames = ['TIME', 'COST', 'HEADWAY', 'TIME_L1',
            'TIME_L2', 'COST_L1', 'COST_L2', 'HEADWAY_L1', 'HEADWAY_L2',
            'GA','SEATS', 'AGE', 'LUGGAGE', 'INCOME', 'MALE', 'WHO',
       'FIRST'] #all explanatory variables to be included in the model
"""
varnames = ['TIME', 'COST', 'HEADWAY','TIME_L1', 'TIME_L2', 'COST_L1', 'COST_L2', 'HEADWAY_L1', 'HEADWAY_L2',
       'SEATS', 'AGE_TRAIN', 'AGE_CAR', 'AGE_SM', 'LUGGAGE_CAR', 'LUGGAGE_SM',
       'LUGGAGE_TRAIN', 'INCOME_CAR', 'INCOME_SM', 'INCOME_TRAIN', 'MALE_CAR',
       'MALE_SM', 'MALE_TRAIN', 'WHO_CAR', 'WHO_SM', 'WHO_TRAIN',
       'FIRST_TRAIN', 'FIRST_SM', 'FIRST_CAR']
"""
asvarnames = ['TIME', 'COST', 'HEADWAY', 'TIME_L1',
            'TIME_L2', 'COST_L1', 'COST_L2', 'HEADWAY_L1', 'HEADWAY_L2'] # alternative-specific variables in varnames
isvarnames = ['GA','SEATS', 'AGE', 'LUGGAGE', 'INCOME', 'MALE', 'WHO','FIRST'] # individual-specific variables in varnames
choice_set=['TRAIN','CAR','SM'] #list of alternatives in the choice set as string
choice_var = df['CHOICE'] # the df column name containing the choice variable
alt_var = df['alt'] # the df column name containing the alternative variable
av = None #df['AV']  #the df column name containing the alternatives' availability
weight_var = None #the df column name containing the weights 
base = 'TRAIN' #reference alternative
R = 200 # number of random draws for estimating mixed logit models
Tol = 1e-4 #Tolerance value for the optimazition routine used in maximum likelihood estimation (default value is 1e-06)
iterations = 200 #number of iterations for the MLE optimization routine (default value is 2,000)
#dist = ['n', 'f'] 
dist = ['n', 'u', 't', 'ln', 'f'] #List of random distributions to select from 
#('n': normal; 'ln': lognormal;'tn': truncated-normal; 'u': uniform; 't': triangular; 'f': fixed)


# ## 5.d. Include modellers' model prerequisites if any

# ### If the modeller wants to prespecify the inclusion of any variables, their coefficient distribution, corrletion or their non-linear transoformation, it can be provided here:

# In[14]:


#pre-included alternative-sepcific variables
psasvar_ind = [0] * len(asvarnames)  #binary indicators representing alternative-specific variables that are prespecified by the user

psisvar_ind = [0] * len(isvarnames) #binary indicators representing individual-specific variables prespecified by the user

#pre-included distributions
#pspecdist_ind = ["f"]* 9 + ["any"] * (len(asvarnames)-9) #variables whose coefficient distribution have been prespecified by the modeller
pspecdist_ind = ["any"] * len(asvarnames)

#prespecification on estimation of intercept
ps_intercept = None  #(True or False or None)

#prespecification on transformations
ps_bctrans = False #(True or False or None)
ps_bcvar_ind = [0] * len(asvarnames) #indicators representing variables with prespecified transformation by the modeller

#prespecification on estimation of correlation
ps_cor = False #(True or False or None)
ps_corvar_ind = [0] * len(asvarnames)  #[1,1,1,1,1] indicators representing variables with prespecified correlation by the modeller

##prespecified interactions
ps_interaction = None #(True or False or None)


# # 5.e. Check to confirm that the inputs provided by the modeller do not have mistakes

# In[15]:


## Check lenghts of asvarnames and isvarnames and their corresponding psasvar_ind,psisvar_ind
if len(asvarnames) != len(psasvar_ind) or len(isvarnames) != len(psisvar_ind):
    raise SyntaxError("lenghts of alterative-specific or individual-specific variable lists do not match with their corresponding prespecification indicator lists")
    
## Check lenghts of asvarnames and their corresponding pspecdist_ind
if len(asvarnames) != len(pspecdist_ind):
    raise SyntaxError("lenght of alterative-specific variable list and coefficient distribution prespecification list do not match")
    
## Check lenghts of asvarnames and their corresponding ps_bcvar_ind
if len(asvarnames) != len(ps_bcvar_ind):
    raise SyntaxError("lenght of alterative-specific variable list and non-linear transformation prespecification lists do not match")
    
## Check lenghts of asvarnames and their corresponding ps_corvar_ind
if len(asvarnames) != len(ps_corvar_ind):
    raise SyntaxError("lenght of alterative-specific variables and correlation prespecification lists do not match")

## More than two asvariables need to be prespecified to allow their correlation 
if len([id for id in ps_corvar_ind if id == 1]) == 1:
    raise SyntaxError("more than one variable needed to estimate correlation")

if len([id for id in ps_corvar_ind if id == 1]) > 0:
    if ps_cor is None or False:
        raise SyntaxError("ps_cor needs to be set as True if prespecified vars are allowed to correlate")

if len([id for id in ps_bcvar_ind if id == 1]) > 0:
    if ps_bctrans is None or False:
        raise SyntaxError("ps_bctrans needs to be set as True if prespecified vars are included in tranformation")

## prespecified features for transformation should not be prespecified with correlation
psbcvarind = [i for i, x in enumerate(ps_bcvar_ind) if x == 1]
pscorvarind = [i for i, x in enumerate(ps_corvar_ind) if x == 1] 
bc_corvar_ind = [var for var in psbcvarind if var in pscorvarind]
if bc_corvar_ind:
    raise SyntaxError("asvar cannot be transformed and correlation at the same time")


# # II. Optimization framework 

# ## 6. Define the hyperparamters for the search

# In[16]:


## We use the default hyperparameters used in Xiang, 2014
HMS = 20 #20 #harmony memory size
HMCR_min = 0.1 #0.9 #minimum harmony memory consideration rate
HMCR_max = 0.99 #0.99 #maximum harmony memory consideration rate
PAR_min = 0.25 # 0.8 #min pitch adjustment
PAR_max = 0.9 #0.85 #maximum pitch adjustment
itr_max = 300 #300 #150 maximum number of iterations
v = 0.80 #0.80 #proportion of iterations to improvise harmony. The rest will be for local search
threshold = 15 #15 #threshold to compare new solution with worst solution in memory


# ## 7. Metaheuristic - Improved Global Best Harmony Search

# In[17]:


#Function to identify prespecified features
def prespec_features(ind_psasvar,ind_psisvar,ind_pspecdist,ind_psbcvar,ind_pscorvar):
    """
    Generates lists of features that are predetermined by the modeller for the model development
    Inputs: 
    (1) ind_psasvar - indicator list for prespecified asvars
    (2) ind_psisvar - indicator list for prespecified isvars
    (3) ind_pspecdist - indicator list for vars with prespecified coefficient distribution
    (4) ind_psbcvar - indicator list for vars with prespecified transformation
    (5) ind_pscorvar - indicator list for vars with prespecified correlation  
    """
    #prespecified alternative-specific variables
    ps_asvar_pos = [i for i, x in enumerate(ind_psasvar) if x == 1]
    ps_asvars = [var for var in asvarnames if asvarnames.index(var) in ps_asvar_pos]
    
    #prespecified individual-specific variables
    ps_isvar_pos = [i for i, x in enumerate(ind_psisvar) if x == 1]
    ps_isvars = [var for var in isvarnames if isvarnames.index(var) in ps_isvar_pos]

    ##prespecified coeff distributions for variables
    ps_rvar_ind = dict(zip(asvarnames,ind_pspecdist))
    ps_rvars = {k:v for k,v in ps_rvar_ind.items() if v != "any"}
    
    #prespecified non-linear transformed variables
    ps_bcvar_pos = [i for i, x in enumerate(ind_psbcvar) if x == 1]
    ps_bcvars = [var for var in asvarnames if asvarnames.index(var) in ps_bcvar_pos]

    #prespecified correlated variables
    ps_corvar_pos = [i for i, x in enumerate(ind_pscorvar) if x == 1]
    ps_corvars = [var for var in asvarnames if asvarnames.index(var) in ps_corvar_pos]
    
    return(ps_asvars,ps_isvars,ps_rvars,ps_bcvars,ps_corvars)


# In[18]:


### Pre-specified features
ps_asvars, ps_isvars, ps_rvars, ps_bcvars, ps_corvars = prespec_features(psasvar_ind,psisvar_ind,pspecdist_ind,ps_bcvar_ind,ps_corvar_ind)
#ps_asvars, ps_isvars, ps_rvars, ps_bcvars, ps_corvars


# ## 7.a Initialize Memory

# In[19]:


def avail_features (asvars_ps, isvars_ps, rvars_ps, bcvars_ps, corvars_ps):
    """
    Generates lists of features that are availbale to select from for model development
    Inputs: 
    (1) asvars_ps - list of prespecified asvars
    (2) isvars_ps - list of prespecified isvars
    (3) rvars_ps - list of vars and their prespecified coefficient distribution
    (4) bcvars_ps - list of vars that include prespecified transformation
    (5) corvars_ps - list of vars with prespecified correlation  
    """
    #available alternative-specific variables for selection
    avail_asvars = [var for var in asvarnames if var not in asvars_ps]
    
    #available individual-specific variables for selection
    avail_isvars = [var for var in isvarnames if var not in isvars_ps]
    
    #available variables for coeff distribution selection
    avail_rvars = [var for var in asvarnames if var not in rvars_ps.keys()]
    
    #available alternative-specific variables for transformation
    avail_bcvars = [var for var in asvarnames if var not in bcvars_ps]
    
    #available alternative-specific variables for correlation
    avail_corvars = [var for var in asvarnames if var not in corvars_ps]
    
    return(avail_asvars,avail_isvars,avail_rvars,avail_bcvars,avail_corvars)


# In[20]:


avail_asvars, avail_isvars, avail_rvars, avail_bcvars, avail_corvars = avail_features (ps_asvars, ps_isvars, ps_rvars, ps_bcvars, ps_corvars)
print('avail_features', avail_asvars, avail_isvars, avail_rvars, avail_bcvars, avail_corvars)


# def df_coeff_col(seed,dataframe,names_asvars,choiceset,var_alt):
#     np.random.seed(seed)
#     random_matrix = np.random.randint(1,len(choiceset)+1,(len(choiceset),len(names_asvars)))
#     
#     ## Finding coefficients type (alt-specific or generic) for corresponding variables
#     alt_spec_pos = []
#     for i in range(random_matrix.shape[1]):
#         pos_freq = pd.Series(range(len(random_matrix[:,i]))).groupby(random_matrix[:,i], sort=False).apply(list).tolist()
#         alt_spec_pos.append(pos_freq)
#     
#     for i in range(len(alt_spec_pos)):
#         for j in range(len(alt_spec_pos[i])):
#             for k in range(len(alt_spec_pos[i][j])):
#                 alt_spec_pos[i][j][k] = choiceset[alt_spec_pos[i][j][k]]
# #                 print('alt_spec_pos[i][j][k]', alt_spec_pos[i][j][k])
# 
# 
#     ## creating dummy columns based on the coefficient type
#     asvars_new = []
#     for i in range(len(alt_spec_pos)):
#         for j in range(len(alt_spec_pos[i])):
#             if len(alt_spec_pos[i][j]) < len(choiceset):
#                 alt_spec_pos_str = [str(integer) for integer in alt_spec_pos[i][j]]
#                 dataframe[names_asvars[i] + '_' +'_'.join(alt_spec_pos_str)] = dataframe[names_asvars[i]] * np.isin(var_alt,alt_spec_pos[i][j])
#                 asvars_new.append(names_asvars[i] + '_' +'_'.join(alt_spec_pos_str))
#             else:
#                 asvars_new.append(names_asvars[i])
#     return(asvars_new)    

# In[21]:


## New function
def df_coeff_col(seed,dataframe,names_asvars,names_isvars,choiceset,var_alt):
    """
    This function creates dummy dataframe columns for variables, which are randomly slected 
    to be estimated with alternative-specific coefficients. 
    Inputs: random seed - int
            dataframe - pd.dataframe
            asvars - list of variable names to be considered
            choise_set - list of available alternatives
            var_alt - dataframe column consisting of alternative variable
    Output: List of as variables considered for model development
    """
    np.random.seed(seed)
    random_matrix = np.random.randint(0,2,len(names_asvars))
    asvars_new = []
    alt_spec_pos_str = [str(var) for var in names_asvars if random_matrix[names_asvars.index(var)] ==1]
    for i in alt_spec_pos_str:
        for j in choiceset:
            dataframe[i + '_' + j] = dataframe[i]*(var_alt == j)
            asvars_new.append(i + '_' + j)
            #print("asvars_new",asvars_new)
    asvars_new.extend([str(integer) for integer in names_asvars if random_matrix[names_asvars.index(integer)] ==0])
    #print ("features after df_coeff_col", asvars_new)
    
    
    ##create interaction variables
    interaction_asvars = np.random.choice(asvars_new,1)
    interaction_isvars = np.random.choice(names_isvars,1)
    new_interaction_varname = interaction_isvars[0] + "_" + interaction_asvars[0]
    asvars_new.append(new_interaction_varname)
    dataframe[new_interaction_varname] = dataframe[interaction_asvars[0]]*dataframe[interaction_isvars[0]]
    
    ##Remove redundant isvar and asvar
    asvars_new = [var for var in asvars_new if var not in interaction_asvars]
    isvars_new = [var for var in names_isvars if var not in interaction_isvars]
    
    return(asvars_new,isvars_new)


# In[22]:


asvars_new, isvars_new = df_coeff_col(5,df,avail_asvars,avail_isvars,choice_set,alt_var)


# In[23]:


asvars_new, isvars_new


# def df_coeff_col(seed,dataframe,names_asvars,choiceset,var_alt):
#     np.random.seed(seed)
#     random_matrix = np.random.randint(0,2,len(names_asvars))
#     asvars_new = []
#     alt_spec_pos_str = [str(var) for var in names_asvars if random_matrix[names_asvars.index(var)] ==1]
#     for i in alt_spec_pos_str:
#         for j in choiceset:
#             dataframe[i + '_' + j] = dataframe[i]*(var_alt == j)
#             
#             asvars_new.append(i + '_' + j)
#     asvars_new.extend([str(integer) for integer in names_asvars if random_matrix[names_asvars.index(integer)] ==0])
#     #print ("features after df_coeff_col", asvars_new)
#     
#     return(asvars_new)

# def remove_redundant_asvars(asvar_list,transasvars,seed):
#     np.random.seed(seed)
#     redundant_asvars = [s for s in asvar_list if any(xs in s for xs in transasvars)]
#     unique_asvars = [var for var in asvar_list if var not in redundant_asvars]
#     select_asvars = []
#     for var in transasvars:
#         re_asvars = [s for s in redundant_asvars if var in s]
#         if re_asvars:
#             select_asvars.append(np.random.choice(re_asvars))
#     final_asvars = sorted(list(set(unique_asvars+select_asvars))) #TODO: think here is the issue
#     return(final_asvars)

# In[24]:


## Removing redundancy if the same variable is included in the model with and without transformation 
#or with a combination of alt-spec and generic coefficients
def remove_redundant_asvars(asvar_list,transasvars,seed):
    redundant_asvars = [s for s in asvar_list if any(xs in s for xs in transasvars)]
    unique_vars = [var for var in asvar_list if var not in redundant_asvars]
    np.random.seed(seed)
    ## When transformations are not applied, the redundancy is created if a variable has both generic & alt-spec co-effs
    if len(transasvars) == 0:
        #print("no trans")
        gen_var_select = [var for var in asvar_list if var in asvarnames]
        #print(gen_var_select)
        alspec_final = [var for var in asvar_list if var not in gen_var_select]
        #print(alspec_final)
        
    else:
        #print("trans")
        gen_var_select = []
        alspec_final = []
        for var in transasvars:
            redun_vars = [item for item in asvar_list if var in item]
            gen_var = [var for var in redun_vars if var in asvarnames]
            if gen_var:
                gen_var_select.append(np.random.choice(gen_var))
            alspec_redun_vars = [item for item in asvar_list if var in item and item not in asvarnames]
            trans_alspec = [i for i in alspec_redun_vars if any(l for l in boxc_l if l in i)]
            lin_alspec = [var for var in alspec_redun_vars if var not in trans_alspec]
            #np.random.seed(seed)
            if np.random.randint(2):
                alspec_final.extend(lin_alspec)
            else:
                alspec_final.extend(trans_alspec)
    np.random.seed(seed)
    if len(gen_var_select) and len(alspec_final) != 0:
        if np.random.randint(2):
            #print("genvars selected_1")
            final_asvars = gen_var_select
            final_asvars.extend(unique_vars)
        else:
            #print("alspec selected_1")
            final_asvars = alspec_final
            final_asvars.extend(unique_vars)
            
    elif len(gen_var_select) != 0:
        #print("genvars selected_2")
        final_asvars = gen_var_select
        final_asvars.extend(unique_vars)
        
    else:
        #print("alspec selected_2")
        final_asvars = alspec_final
        final_asvars.extend(unique_vars)
    
    
    return(list(dict.fromkeys(final_asvars)))


# In[25]:


remove_redundant_asvars(asvars_new,trans_asvars,3)


# In[26]:


def generate_sol(data,seed,asvars_avail, isvars_avail, rvars_avail, transasvars, bcvars_avail,corvars_avail,asvars_ps, isvars_ps, rvars_ps, 
                 bcvars_ps, corvars_ps, bctrans_ps, cor_ps, intercept_ps):
    """
    Generates list of random model features and then includes modeller prespecifications
    Inputs:
    (1) seed - seed for random generators
    (2) asvars_avail - list of available asvars for random selection
    (3) isvars_avail - list of available isvars for random selection
    (4) rvars_avail - list of available vars for randomly selected coefficient distribution
    (5) bcvars_avail - list of available vars for random selection of transformation
    (6) corvars_avail - list of available vars for random selection of correlation 
    ## Prespecification information
    (1) asvars_ps - list of prespecified asvars
    (2) isvars_ps - list of prespecified isvars
    (3) rvars_ps - list of vars and their prespecified coefficient distribution
    (4) bcvars_ps - list of vars that include prespecified transformation
    (5) corvars_ps - list of vars with prespecified correlation  
    (6) bctrans_ps - prespecified transformation boolean
    (7) cor_ps - prespecified correlation boolean
    (8) intercept_ps - prespecified intercept boolean
    
    """

    np.random.seed(seed)
    ind_availasvar = []
    for i in range(len(asvars_avail)):
        ind_availasvar.append(np.random.randint(2))
    asvar_select_pos = [i for i, x in enumerate(ind_availasvar) if x == 1]
    asvars_1 = [var for var in asvars_avail if asvars_avail.index(var) in asvar_select_pos]
    asvars_1.extend(asvars_ps)

    #print(asvars_1)
    
    asvars_new = remove_redundant_asvars(asvars_1,transasvars,seed)
    #print(asvars_new)
    #asvars = df_coeff_col(seed,data,asvars_new,choice_set,alt_var)
    
    #print(asvars)
   
    ind_availisvar = []
    for i in range(len(isvars_avail)):
        ind_availisvar.append(np.random.randint(2))
    isvar_select_pos = [i for i, x in enumerate(ind_availisvar) if x == 1]
    isvars = [var for var in isvars_avail if isvars_avail.index(var) in isvar_select_pos]
    isvars.extend(isvars_ps)

    asvars, isvars = df_coeff_col(seed,data,asvars_new,isvars,choice_set,alt_var)
    
    r_dist = []
    avail_rvar = [var for var in asvars if var in rvars_avail]
    for i in range(len(avail_rvar)):
        r_dist.append(np.random.choice(dist))
        
    
    rvars = dict(zip(avail_rvar,r_dist))
    rvars.update(rvars_ps)
    rand_vars = {k:v for k,v in rvars.items() if v!="f" and k in asvars}
    r_dis = [dis for dis in dist if dis != "f"]
    for var in corvars_ps:
        if var in asvars and var not in rand_vars.keys():
            rand_vars.update({var:np.random.choice(r_dis)})
    
    
    if bctrans_ps is None:
        bctrans = bool(np.random.randint(2, size=1))
    else:
        bctrans = bctrans_ps
        
    if bctrans:
        ind_availbcvar = []
        for i in range(len(bcvars_avail)):
            ind_availbcvar.append(np.random.randint(2))
        bcvar_select_pos = [i for i, x in enumerate(ind_availbcvar) if x == 1]
        bcvars = [var for var in bcvars_avail if bcvars_avail.index(var) in bcvar_select_pos]
        bcvars.extend(bcvars_ps)
        bc_vars = [var for var in bcvars if var in asvars and var not in corvars_ps]
    else:
        bc_vars = []
        
    if cor_ps is None:
        cor = bool(np.random.randint(2, size=1))
    else:
        cor = cor_ps
    
    if cor:
        ind_availcorvar = []
        for i in range(len(corvars_avail)):
            ind_availcorvar.append(np.random.randint(2))
        corvar_select_pos = [i for i, x in enumerate(ind_availcorvar) if x == 1]
        corvars = [var for var in corvars_avail if corvars_avail.index(var) in corvar_select_pos]
        corvars.extend(corvars_ps)
        cor_vars = [var for var in corvars if var in rand_vars.keys() and var not in bc_vars]
        if len(cor_vars) <2:
            cor = False
            cor_vars = []
    else:
        cor_vars = []
        
    if intercept_ps is None:
        asc_ind = bool(np.random.randint(2, size=1))
    else:
        asc_ind = intercept_ps
    return(asvars,isvars,rand_vars,bc_vars,cor_vars,bctrans,cor,asc_ind)


# In[27]:


generate_sol(df,2,avail_asvars, avail_isvars, avail_rvars, trans_asvars, avail_bcvars,avail_corvars,ps_asvars, ps_isvars, ps_rvars, 
                 ps_bcvars, ps_corvars, ps_bctrans, ps_cor, ps_intercept)


# In[28]:


#Fit multinomial logit model
def fit_mnl(dat,as_vars,is_vars,bcvars,choice,alt,id_choice,asc_ind):
    
    """
    Estimates multinomial model for the generated solution
    Inputs:
    (1) dat in csv
    (2) as_vars: list of alternative-specific variables
    (3) is_vars: list of individual-specific variables
    (4) bcvars: list of box-cox variables
    (5) choice: df column with choice variable
    (6) alt: df column with alternative variables
    (7) id_choice: df column with choice situation id
    (8) asc_ind: boolean for fit_intercept
    
    """
    data = dat.copy()
    all_vars = as_vars + is_vars
    #print("all_vars inputs for mnl",all_vars)
    X = data[all_vars]
    y = choice
    model = MultinomialLogit()
    model.fit(X, y, varnames=all_vars, isvars=is_vars, alts=alt_var, ids=id_choice, fit_intercept=asc_ind,transformation="boxcox",transvars = bcvars,maxiter=iterations, gtol = Tol,avail = av,weights=weight_var)
    rand_vars = {}
    cor_vars = []
    bc_vars = [var for var in bcvars if var not in isvarnames]
    print(model.summary())
    return(model.bic,as_vars,is_vars,rand_vars,bc_vars,cor_vars,model.convergence,model.pvalues,model.coeff_names)


# In[29]:


def fit_mxl(dat,as_vars,is_vars,rand_vars,bcvars,corvars,choice,alt,id_choice,id_val,asc_ind):
    """
    Estimates the model for the generated solution
    Inputs:
    (1) dat: dataframe in csv
    (2) as_vars: list of alternative-specific variables
    (3) is_vars: list of individual-specific variables
    (4) bcvars: list of box-cox variables
    (5) choice: df column with choice variable
    (6) corvars: list of variables allowed to correlate
    (7) alt: df column with alternative variables
    (8) id_choice: df column with choice situation id
    (9) id_val: df column with individual id
    (10) asc_ind: boolean for fit_intercept
    
    """
    data = dat.copy()
    all_vars = as_vars + is_vars
    X = data[all_vars]
    y = choice
    if corvars == []:
        corr = False
    else:
        corr = corvars
    bcvars = [var for var in bcvars if var not in isvarnames]
    model = MixedLogit()
    model.fit(X, y, varnames=all_vars, alts=alt_var, isvars=is_vars, 
                ids=id_choice, panels=id_val,
                randvars=rand_vars,n_draws=R,
                fit_intercept = asc_ind,correlation = corvars, transformation = "boxcox", 
                transvars = bcvars,maxiter=iterations, avail=av, gtol = Tol,weights=weight_var)
    print(model.summary())
    return(model.bic,as_vars,is_vars,rand_vars,bcvars,corvars,model.convergence,model.pvalues,model.coeff_names)


# In[30]:


def evaluate_objective_function(new_df,seed,as_vars,is_vars,rand_vars,bc_vars,cor_vars,choice,alts,id_choice,id_val,asc_ind):
    """
    (1) Evaluates the objective function (estimates the model and BIC) for a given list of variables (estimates the model coefficeints, LL and BIC)
    (2) If the solution generated in (1) contains statistically insignificant variables, 
    a new model is generated by removing such variables and the model is re-estimated 
    (3) the functions returns estimated solution only if it converges
    Inputs: lists of variable names, individual specific variables, variables with random coefficients, 
    name of the choice variable in df, list of alternatives, choice_id, individual_id(for panel data) and fit intercept bool
    """   
    #print("evaluating OF")
    all_vars = as_vars + is_vars
    
    sol =[10000000.0,[],[],{},[],[],False] 
    convergence = False
    
    #Estimate model if input variables are present in specification
    if all_vars:
        print("features for round 1",as_vars,is_vars,rand_vars,bc_vars,cor_vars,asc_ind)
        if bool(rand_vars):
            print("estimating an MXL model")
            
            bic,asvars,isvars,randvars,bcvars,corvars,conv,sig,coefs = fit_mxl(new_df,as_vars,is_vars,rand_vars,bc_vars,cor_vars,choice,alts,id_choice,id_val,asc_ind)
        
        else:
            print("estimating an MNL model")
            #print(new_df.columns)
            bic,asvars,isvars,randvars,bcvars,corvars,conv,sig,coefs = fit_mnl(new_df,as_vars,is_vars,bc_vars,choice,alts,id_choice,asc_ind)
        if conv:
            print("solution converged in first round")
            sol = [bic,asvars,isvars,randvars,bcvars,corvars,asc_ind] 
            convergence = conv
            if all(v for v in sig <= 0.05):
                print("solution has all sig-values in first  round") 
                return (sol,convergence)
        
            else:
                while any([v for v in sig if v > 0.05]):
                    print("solution consists insignificant coeffs")
                    #create dictionary of {coefficient_names: p_values}
                    p_vals = dict(zip(coefs,sig)) 
                    #print("p_vals =", p_vals)
                    r_dist = [dis for dis in dist if dis!= 'f'] #list of random distributions

                    #create list of variables with insignificant coefficients
                    non_sig = [k for k,v in p_vals.items() if v > 0.05] #list of non-significant coefficient names
                    print("non-sig coeffs are", non_sig)
                    #keep only significant as-variables
                    asvars_round2 = [var for var in asvars if var not in non_sig]  # as-variables with significant p-vals
                    asvars_round2.extend(ps_asvars)
                    print("asvars_round2 for round 2", asvars_round2)
                    #replace non-sig alt-spec coefficient with generic coefficient
                    nsig_altspec = []
                    for var in asvarnames:
                        ns_alspec = [x for x in non_sig if x.startswith(var)]
                        nsig_altspec.extend(ns_alspec)
                        nsig_altspec_vars = [var for var in nsig_altspec if var not in asvarnames]
                    print("nsig_altspec_vars",nsig_altspec_vars)    
                    
                   
                    # Replacing non-significant alternative-specific coeffs with generic coeffs estimation
                    if nsig_altspec_vars:
                        gen_var = []
                        for i in range(len(nsig_altspec_vars)):
                            gen_var.extend(nsig_altspec_vars[i].split("_"))
                        gen_coeff = [var for var in asvarnames if var in gen_var] 
                        if asvars_round2:
                            redund_vars = [s for s in gen_coeff if any(s in xs for xs in asvars_round2)]
                            print("redund_vars for round 2",redund_vars)
                            asvars_round2.extend([var for var in gen_coeff if var not in redund_vars])
                            
                        #rem_asvars = remove_redundant_asvars(asvars_round2,trans_asvars,seed)
                            print("asvars_round2 before removing redundancy", asvars_round2)
                            #rem_asvars = remove_redundant_asvars(asvars_round2,trans_asvars,seed)
                            rem_asvars = sorted(list(set(asvars_round2))) #checking if remove_redundant_asvars is needed or not
                        else:
                            rem_asvars = gen_coeff
    
                    else:
                        rem_asvars = sorted(list(set(asvars_round2)))
                    print("rem_asvars =", rem_asvars)
                    #remove insignificant is-variables
                    ns_isvars = []
                    for isvar in isvarnames:
                        ns_isvar = [x for x in non_sig if x.startswith(isvar)]
                        ns_isvars.extend(ns_isvar)
                    remove_isvars = []
                    for i in range(len(ns_isvars)):
                        remove_isvars.extend(ns_isvars[i].split("."))
                    
                    remove_isvar = [var for var in remove_isvars if var in isvars]
                    most_nsisvar = {x:remove_isvar.count(x) for x in remove_isvar}
                    rem_isvar = [k for k,v in most_nsisvar.items() if v == (len(choice_set)-1)]
                    isvars_round2 = [var for var in is_vars if var not in rem_isvar] # individual specific variables with significant p-vals
                    isvars_round2.extend(ps_isvars)
                    #print("isvars_round2 =", isvars_round2)
                    rem_isvars = sorted(list(set(isvars_round2)))
                    #print("rem_isvars =", rem_isvars)

                    #remove intercept if not significant and not prespecified
                    ns_intercept = [x for x in non_sig if x.startswith('_intercept.')] #non-significant intercepts
                    #print("ns_intercept =", ns_intercept)
                    
                    new_asc_ind = asc_ind
                    
                    if ps_intercept is None:
                        if len(ns_intercept) == len(choice_set)-1:
                            new_asc_ind = False
                    else:
                        new_asc_ind = ps_intercept

                    #print("new_asc_ind =", new_asc_ind)
                    
                    #remove insignificant random coefficients

                    ns_sd = [x for x in non_sig if x.startswith('sd.')] #non-significant standard deviations
                    ns_sdval = [str(i).replace( 'sd.', '') for i in ns_sd] #non-significant random variables
                    #print("ns_sdval =", ns_sdval)
                    remove_rdist = [x for x in ns_sdval if x not in ps_rvars.keys() or x not in rem_asvars]#non-significant random variables that are not pre-included

                    rem_rand_vars = {k:v for k, v in randvars.items() if k in rem_asvars and k not in remove_rdist}#random coefficients for significant variables 
                    rem_rand_vars.update({k:v for k,v in ps_rvars.items() if k in rem_asvars and v!='f'})
                    print("rem_rand_vars =", rem_rand_vars)
                    ## including ps_corvars in the model if they are included in rem_asvars
                    for var in ps_corvars:
                        if var in rem_asvars and var not in rem_rand_vars.keys():
                            rem_rand_vars.update({var:np.random.choice(r_dist)})
                    #print("rem_rand_vars =", rem_rand_vars)
                    #remove transformation if not significant and non prespecified
                    ns_lambda = [x for x in non_sig if x.startswith('lambda.')] #insignificant transformation coefficient
                    ns_bctransvar = [str(i).replace( 'lambda.', '') for i in ns_lambda] #non-significant transformed var
                    rem_bcvars = [var for var in bcvars if var in rem_asvars and var not in ns_bctransvar and var not in ps_corvars]
                    #print("rem_bcvars =", rem_bcvars)

                    #remove insignificant correlation
                    ns_chol = [x for x in non_sig if x.startswith('chol.')] #insignificant cholesky factor
                    ns_cors = [str(i).replace( 'chol.', '') for i in ns_chol] #insignicant correlated variables
                    #create a list of variables whose correlation coefficient is insignificant
                    if ns_cors:
                        ns_corvar = []
                        for i in range(len(ns_cors)):
                            ns_corvar.extend(ns_cors[i].split("."))
                        most_nscorvars = {x:ns_corvar.count(x) for x in ns_corvar}
                        print(most_nscorvars)
                        #check frequnecy of variable names in non-significant coefficients
                        nscorvars = [k for k,v in most_nscorvars.items() if v >= int(len(corvars)*0.75)]
                        print (nscorvars)
                        nonps_nscorvars = [var for var in nscorvars if var not in ps_corvars]
                        #if any variable has insignificant correlation with all other variables, their correlation is removed from the solution
                        if nonps_nscorvars:
                        #list of variables allowed to correlate
                            rem_corvars = [var for var in rem_rand_vars.keys() if var not in nonps_nscorvars and var not in rem_bcvars]
                        else:
                            rem_corvars = [var for var in rem_rand_vars.keys() if var not in rem_bcvars]

                        #need atleast two variables in the list to estimate correlation coefficients
                        if len(rem_corvars)<2:
                            rem_corvars = []
                    else:
                        rem_corvars = [var for var in corvars if var in rem_rand_vars.keys() and var not in rem_bcvars]
                        if len(rem_corvars)<2:
                            rem_corvars = []
                    #print("rem_corvars =", rem_corvars)
                    
                    #Evaluate objective function with significant feautures from round 1
                    #print("features for round2",rem_asvars,rem_isvars,rem_rand_vars,rem_bcvars,rem_corvars,new_asc_ind)
                                            
                    rem_alvars = rem_asvars + rem_isvars
                    if rem_alvars:
                        #print("remaining vars present")
                        if set(rem_alvars) != set(all_vars) or set(rem_rand_vars) != set(rand_vars) or set(rem_bcvars) != set(bcvars) or set(rem_corvars) != set(corvars) or new_asc_ind !=asc_ind:
                            print("not same as round 1 model")
                            
                        else:
                            print("model 2 same as round 1 model")
                            return(sol,convergence)
                        
                        if bool(rem_rand_vars):
                            print("MXL model round 2")
                            bic,asvars,isvars,randvars,bcvars,corvars,conv,sig,coefs = fit_mxl(new_df,rem_asvars,rem_isvars,rem_rand_vars,rem_bcvars,rem_corvars,choice,alts,id_choice,id_val,new_asc_ind)                         
                        else:
                            print("MNL model round 2")
                            bic,asvars,isvars,randvars,bcvars,corvars,conv,sig,coefs = fit_mnl(new_df,rem_asvars,rem_isvars,rem_bcvars,choice,alts,id_choice,new_asc_ind)

                        #print(sol)
                        if conv:
                            sol = [bic,asvars,isvars,randvars,bcvars,corvars,new_asc_ind]
                            convergence = conv
                            if all([v for v in sig if v <= 0.05]):
                                break
                                #return(sol,convergence)
                            #if only some correlation coefficients or intercept values are insignificant, we accept the solution
                            p_vals = dict(zip(coefs,sig)) 
                            non_sig = [k for k,v in p_vals.items() if v > 0.05]
                            print("non_sig in round 2", non_sig)
                            
                            sol[1] = [var for var in sol[1] if var not in non_sig or var in ps_asvars] #keep only significant vars
                            
                            ##Update other features of solution based on sol[1]
                            sol[3] = {k:v for k,v in sol[3].items() if k in sol[1]}
                            sol[4] = [var for var in sol[4] if var in sol[1] and var not in ps_corvars]
                            sol[5] = [var for var in sol[5] if var in sol[3].keys and var not in sol[4]]
                                      
                            ## fit_intercept = False if all intercepts are insignificant
                            if len([var for var in non_sig if var in ['_intercept.' + var for var in choice_set]])== len(non_sig):
                                    if len(non_sig) == len(choice_set)-1:
                                        sol[-1] = False
                                        return(sol,convergence)
                            
                            all_ns_int = [x for x in non_sig if x.startswith('_intercept.')]
                            all_ns_cors = [x for x in non_sig if x.startswith('chol.')]
                            
                            all_ns_isvars = []
                            for isvar in isvarnames:
                                ns_isvar = [x for x in non_sig if x.startswith(isvar)]
                                all_ns_isvars.extend(ns_isvar)
                            
                            irrem_nsvars = all_ns_isvars + all_ns_int + all_ns_cors
                            if all(nsv in irrem_nsvars for nsv in non_sig):
                                print("non-significant terms cannot be further eliminated")
                                return(sol,convergence)
                            
                            if non_sig == all_ns_cors or non_sig == all_ns_int or non_sig == list(set().union(all_ns_cors, all_ns_int)) :
                                print("only correlation coefficients or intercepts are insignificant")
                                return(sol,convergence)
                        
                            if all([var in ps_asvars or var in ps_isvars or var in ps_rvars.keys() for var in non_sig]):
                                print("non-significant terms are pre-specified")
                                return(sol,convergence)
                            
                            if len([var for var in non_sig if var in ['sd.' + var for var in ps_rvars.keys()]]) == len(non_sig):
                                print("non-significant terms are pre-specified random coefficients")
                                return(sol,convergence)
                        
                        else:
                            #convergence = False
                            print("convergence not reached in round 2 so final sol is from round 1")
                            return(sol,convergence)
                    else:
                        print("no vars for round 2")
                        return(sol,convergence)
        else:
            convergence = False
            print("convergence not reached in round 1")
            return(sol,convergence)
    else:
        print("no vars when function called first time")
    return(sol,convergence)


# evaluate_objective_function(df,1,as_vars,is_vars,rand_vars,bc_vars,cor_vars,choice_var,alt_var,choice_id,ind_id,True)

# In[31]:


## Set Random Seed
global_seed = 1609
np.random.seed(global_seed)
seeds = np.random.choice(50000, 23000, replace = False)


# In[32]:


#Initialize harmony memory and opposite harmony memory of size HMS with random slutions
def initialize_memory(choice_data,HM_size,asvars_avail, isvars_avail, rvars_avail, bcvars_avail,
                      corvars_avail,asvars_ps, isvars_ps, rvars_ps, bcvars_ps,
                      corvars_ps,bctrans_ps,cor_ps,intercept_ps):
    
    """
    Creates two lists (called the harmony memory and opposite harmony memory) 
    harmony memory - containing the initial randomly generated solutions 
    opposite harmony memory - containing random solutions that include variables not included in harmony memory
    Inputs: harmony memory size (int), all variable names, individual-specific variable, prespecifications provided by user
    """
    init_HM =  code_name + 'initialize_memory_' + current_date + '.txt'
    sys.stdout = open(init_HM,'wt')
    
    
    #HM_sol_labels = create_sol_labels(1,HM_size+1)
    #OHM_sol_labels = create_sol_labels(HMS+1,(HMS*2)+1)
    
    #set random seeds
    
    HM = []
    opp_HM = []
    
    HM.append(base_model)
    #print("HM with base model is",HM)
    
    #Add an MXL with full covriance structure
    
    #Create initial harmony memory
    unique_HM = []
    for i in range(len(seeds)):
        seed = seeds[i]
        asvars,isvars,randvars,bcvars,corvars,bctrans,cor,asconstant = generate_sol(choice_data,seed,asvars_avail, isvars_avail, rvars_avail,
                                                                                    trans_asvars,bcvars_avail, corvars_avail,asvars_ps, isvars_ps, rvars_ps, bcvars_ps,
                                                                                    corvars_ps,bctrans_ps,cor_ps,intercept_ps)
        
        sol,conv = evaluate_objective_function(choice_data,seed,asvars,isvars,randvars,
                                               bcvars,corvars,choice_var,alt_var,choice_id,ind_id,asconstant)
        #print("new HM solution is",sol,conv)
       
        #print("convergence for HM sol is",conv,sol)
        if conv:
        #add to memory
        #Har_Mem = dict(zip(HM_sol_labels, HM))
            # Similarity check to keep only unique solutions in harmony memory
            if len(HM) > 0: # only do check if there are already solutions
                bic_list = [hm_sol[0] for hm_sol in HM]
                discrepancy = 0.1 * min(bic_list)    # TODO: arbitrary choice ... improve?

                unique_HM_discrepancy = []
                for sol_hm in HM:
                    if np.abs(sol_hm[0] - sol_hm[0]) <= discrepancy:
                        unique_HM_discrepancy.append(sol_hm)

                if len(unique_HM_discrepancy) > 0:
                    # check if varnames, randvars, bcvars, corrvars and fit are the same as similar BIC solns
                    # if 2 or more are same then do not accept solution
                    hm_varnames = [sol_hm[1] for sol_hm in unique_HM_discrepancy]
                    hm_randnames = [sol_hm[2] for sol_hm in unique_HM_discrepancy]
                    hm_trans = [sol_hm[3] for sol_hm in unique_HM_discrepancy]
                    hm_correlation = [sol_hm[4] for sol_hm in unique_HM_discrepancy]
                    hm_intercept = [sol_hm[5] for sol_hm in unique_HM_discrepancy]

                    similarities = 0
                    if sol[0] in hm_varnames:
                        similarities += 1

                    if sol[1] in hm_randnames:
                        similarities += 1

                    if sol[2] in hm_trans:
                        similarities += 1

                    if sol[3] in hm_correlation:
                        similarities += 1

                    if sol[4] in hm_intercept:
                        similarities += 1

                    if similarities > 3: # accepts solution if 2 or more aspects of solution are different
                        conv = False # make false so solution isn't added
        if conv:
            HM.append(sol)
            #print("new harmony is", HM)
            #keep only unique solutions in memory
            used = set()
            unique_HM = [used.add(tuple(x[:1])) or x for x in HM if tuple(x[:1]) not in used]
            unique_HM = sorted(unique_HM, key = lambda x: x[0])
            print("harmony memory for iteration", i, "is", unique_HM)
        
        print("estimating opposite harmony memory")
        #if len(unique_HM) == HMS:  
        
        #create opposite harmony memory with variables that were not included in the harmony memory's solution
        
        #list of variables that were not present in previously generated solution for HM
        ad_var = [x for x in varnames if x not in sol[1]] 
        seed = seeds[i+HMS]
        op_asvars, op_isvars, op_rvars, op_bcvars,op_corvars,op_bctrans,op_cor,op_asconstant = generate_sol(choice_data,seed,asvars_avail, isvars_avail, rvars_avail,
                                                                                    trans_asvars,bcvars_avail, corvars_avail,asvars_ps, isvars_ps, rvars_ps, bcvars_ps,
                                                                                    corvars_ps,bctrans_ps,cor_ps,intercept_ps)
        
        #evaluate objective function of opposite solution
        print("opp sol features",op_asvars, op_isvars, op_rvars, op_bcvars,op_corvars,op_bctrans,op_cor,op_asconstant)
        opp_sol, opp_conv = evaluate_objective_function(choice_data,seed,op_asvars, op_isvars, op_rvars, op_bcvars,op_corvars,
                                                        choice_var,alt_var,choice_id,ind_id,op_asconstant) 
        if opp_conv:
            # Similarity check to keep only unique solutions in opposite harmony memory
            if len(opp_HM) > 0: # only do check if there are already solutions
                bic_list = [sol[0] for sol in unique_opp_HM]
                discrepancy = 0.1 * min(bic_list)    # TODO: arbitrary choice ... improve?

                unique_opp_HM_discrepancy = []
                for opp in opp_HM:
                    if np.abs(opp[0] - opp_sol[0]) <= discrepancy:
                        unique_opp_HM_discrepancy.append(opp)

                if len(unique_opp_HM_discrepancy) > 0:
                    # check if varnames, randvars, bcvars, corrvars and fit are the same as similar BIC solns
                    # if 2 or more are same then do not accept solution
                    opp_HM_varnames = [sol[1] for sol in unique_opp_HM_discrepancy]
                    opp_HM_randnames = [sol[2] for sol in unique_opp_HM_discrepancy]
                    opp_HM_trans = [sol[3] for sol in unique_opp_HM_discrepancy]
                    opp_HM_correlation = [sol[4] for sol in unique_opp_HM_discrepancy]
                    opp_HM_intercept = [sol[5] for sol in unique_opp_HM_discrepancy]

                    similarities = 0
                    if opp_sol[0] in opp_HM_varnames:
                        similarities += 1

                    if opp_sol[1] in opp_HM_randnames:
                        similarities += 1

                    if opp_sol[2] in opp_HM_trans:
                        similarities += 1

                    if opp_sol[3] in opp_HM_correlation:
                        similarities += 1

                    if opp_sol[4] in opp_HM_intercept:
                        similarities += 1

                    if similarities > 3: # accepts solution if 2 or more aspects of solution are different
                        opp_conv = False # make false so solution isn't added
            
        if opp_conv:
            opp_HM.append(opp_sol)
            opp_used = set()
            unique_opp_HM = [opp_used.add(tuple(x[:1])) or x for x in opp_HM if tuple(x[:1]) not in opp_used]
            unique_opp_HM = sorted(unique_opp_HM, key = lambda x: x[0])            
            print("unique_opp_HM is for iteration", i, "is", unique_opp_HM)
#             print("len(unique_opp_HM)", len(unique_opp_HM))
            if len(unique_opp_HM) == HMS:
                #if len(unique_opp_HM) == HMS:
                    break
            sys.stdout.flush()
    return(unique_HM,unique_opp_HM)
##We need to make sure that the BICs of solutions in harmony memory are different from each other by atleast the throshold value


# In[33]:


HM, O_HM = initialize_memory(df,HMS,avail_asvars, avail_isvars, avail_rvars, avail_bcvars,
                      avail_corvars,ps_asvars, ps_isvars, ps_rvars, ps_bcvars,
                      ps_corvars,ps_bctrans,ps_cor,ps_intercept)


# In[34]:


## Combine both harmonies 
Init_HM = HM + O_HM 

##Remove duplicate solutions if present
unique = set()
unique_HM = [unique.add(tuple(x[:1])) or x for x in Init_HM if tuple(x[:1]) not in unique]

## Sort unique harmony memory from min.BIC to max. BIC
HM_sorted = sorted(unique_HM, key = lambda x: x[0])

## Trim the Harmony memory's size as per the harmony memory size
HM = HM_sorted[:HMS]


# In[35]:


hm = HM.copy()


# ## 7.b Improvise harmony 

# ## 7.b.i Harmony Consideration

# In[36]:


def harmony_consideration(har_mem,HMCR_itr,seeds,itr):
    seed = seeds[HMS*2+itr]
    """
    If a generated random number is less than or equal to the harmony memory consideration rate (HMCR)
    then 90% of a solution already in memory will be randomly selected to build the new solution.
    Else a completely new random solution is generated
    Inputs: harmony memory, HMCR for the current interation, random seeds, iteration number
    """
    new_sol = []

    if  np.random.choice([0,1], p=[1-HMCR_itr,HMCR_itr]) <= HMCR_itr:
        print("harmony consideration")
        m_pos = np.random.choice(len(har_mem)) #randomly choose the position of any one solution in harmony memory
        select_new_asvars_index = np.random.choice([0,1],size = len(HM[m_pos][1]), p = [1-HMCR_itr, HMCR_itr])
        select_new_asvars = [i for (i, v) in zip(HM[m_pos][1], select_new_asvars_index) if v] 
        select_new_asvars = list(np.random.choice(har_mem[m_pos][1],int((len(har_mem[m_pos][1]))*HMCR_itr),replace = False)) #randomly select 90% of the variables from solution at position m_pos in harmony memory
        n_asvars = sorted(list(set().union(select_new_asvars, ps_asvars)))
        new_asvars = remove_redundant_asvars(n_asvars,trans_asvars,seed)
        new_sol.append(new_asvars)
        print("new_asvars",new_asvars)
        
        select_new_isvars_index = np.random.choice([0,1],size = len(HM[m_pos][2]), p = [1-HMCR_itr, HMCR_itr])
        select_new_isvars = [i for (i, v) in zip(HM[m_pos][2], select_new_isvars_index) if v] 
        #select_new_isvars = list(np.random.choice(har_mem[m_pos][2],int((len(har_mem[m_pos][2]))*HMCR_itr),replace = False, p=[1-HMCR_itr, HMCR_itr]))
        new_isvars = sorted(list(set().union(select_new_isvars, ps_isvars)))
        print("new_isvars",new_isvars)
        new_sol.append(new_isvars)
        
        #include distributions for the variables in new solution based on the solution at m_pos in memory
        r_pos = {k: v for k, v in har_mem[m_pos][3].items() if k in new_asvars}
        print("r_pos",r_pos)
        new_sol.append(r_pos)
        
        
        #if no prespecified regarding bc-transformation, randomly choose whether to apply a transformation    
        """
        if ps_bctrans is None:
            bc_trans = bool(np.random.randint(2, size=1))
        else:
            bc_trans = ps_bctrans
        
        if bc_trans:
            new_bcvars = list(np.random.choice(new_asvars,int(len(new_asvars)*np.random.rand(1)), replace=False)) #random choice for bc transformation
            ps_bcasvar = [var for var in ps_bcvars if var in new_asvars]
            bcvars_new = sorted(list(set().union(new_bcvars,ps_bcasvar)))
            bcvars =  [var for var in bcvars_new if var not in ps_corvars] #remove those with pre-specified correlation
        else:
            bcvars = []
        
        """
        
        new_bcvars = [var for var in har_mem[m_pos][4] if var in new_asvars and var not in ps_corvars]
        new_sol.append(new_bcvars)
        
        #include correlation in solution 
        """
        if ps_cor is None:
            new_corr = bool(np.random.randint(2, size=1))
        else:
            new_corr = ps_cor
        if new_corr:
            new_corvars = [x for x in r_pos.keys() if x not in bcvars]
        else:
            new_corvars = []
        #at least two correlated variables are required
        if len(new_corvars)<2:
            new_corvars = []
        new_sol.append(new_corvars)
        """
        new_corvars = [var for var in har_mem[m_pos][5] if var in r_pos.keys() and var not in new_bcvars]
        new_sol.append(new_corvars)
        
        #Take fit_intercept from m_pos solution in memory
        intercept = har_mem[m_pos][6]
        new_sol.append(intercept)
        print("new sol after HMC-1", new_sol)
    else:
        print("harmony not considered")
        #if harmony memory consideration is not conducted, then a new solution is generated
                
        asvars,isvars,randvars,bcvars,corvars,bctrans,cor,asconstant = generate_sol(df,seed,avail_asvars, avail_isvars, avail_rvars,trans_asvars,
                                                                                    avail_bcvars, avail_corvars,ps_asvars, ps_isvars, ps_rvars, ps_bcvars,
                                                                                    ps_corvars,ps_bctrans,ps_cor,ps_intercept)                  
        new_sol = [asvars,isvars,randvars,bcvars,corvars,asconstant]
        print("new sol after HMC-2", new_sol)
    return(new_sol)


# ## 7.b.ii Pitch Adjustment

# ### Following functions are used to conduct pitch adjustment on the given solution

# In[37]:


def add_new_asfeature(solution,seed):
    """
    Randomly selects an as variable, which is not already in solution
    Inputs: solution list contianing all features generated from harmony consideration
    ##TODO: Include alternative-specific coefficients
    """
    new_asvar = [var for var in asvarnames if var not in solution[0]]
    print('new_asvar',new_asvar)
    if new_asvar:
        n_asvar = list(np.random.choice(new_asvar,1))
        solution[0].extend(n_asvar)
        solution[0] = remove_redundant_asvars(solution[0],trans_asvars,seed)
        solution[0] = sorted(list(set(solution[0])))
        print("new sol",solution[0])
        
        dis = []
        r_vars = {}
        for i in solution[0]:
            if i in solution[2].keys():
                r_vars.update({k:v for k,v in solution[2].items() if k == i})
                print("r_vars", r_vars)
            else:
                if i in ps_rvars.keys():
                    r_vars.update({i:ps_rvars[i]})
                    print("r_vars", r_vars)
                else:
                    r_vars.update({i:np.random.choice(dist)})
                    print("r_vars", r_vars)
        solution[2] = {k:v for k,v in r_vars.items() if k in solution[0] and v!= 'f'}
                    
    solution[4] = [var for var in solution[4] if var in solution[2].keys() and var not in solution[3]]
    
    if ps_intercept is None:
        solution[5] = bool(np.random.randint(2))
    print(solution)
    return(solution)


# In[38]:


def add_new_isfeature(solution):
    """
    Randomly selects an is variable, which is not already in solution
    Inputs: solution list contianing all features generated from harmony consideration
    """
    if solution[1]:
        new_isvar = [var for var in isvarnames if var not in solution[1]]
        if new_isvar:
            n_isvar = list(np.random.choice(new_isvar,1))
            solution[1] = sorted(list(set(solution[1]).union(n_isvar)))
    return(solution)


# In[39]:


def add_new_bcfeature(solution):
    """
    Randomly selects a variable to be transformed, which is not already in solution
    Inputs: solution list contianing all features generated from harmony consideration
    """
    if ps_bctrans == None:
        bctrans = bool(np.random.randint(2, size=1))
    else:
        bctrans = ps_bctrans
    if bctrans:
        new_bcvar = [var for var in solution[0] if var not in ps_corvars]
        solution[3] = sorted(list(set(solution[3]).union(new_bcvar)))
    else:
        solution[3] = []
    solution[4] = [var for var in solution[4] if var not in solution[3]]
    return(solution)


# In[40]:


def add_new_corfeature(solution):
    """
    Randomly selects variables to be correlated, which is not already in solution
    Inputs: solution list contianing all features generated from harmony consideration
    """
    if ps_cor == None:
        cor = bool(np.random.randint(2, size=1))
    else:
        cor = ps_cor
    if cor:
        new_corvar = [var for var in solution[2].keys() if var not in solution[3]]
        solution[4] = sorted(list(set(solution[4]).union(new_corvar)))
    else:
        solution[4] = []
    if len(solution[4]) < 2:
        solution[4] = []
    solution[3] = [var for var in solution[3] if var not in solution[4]]
    return(solution)


# In[41]:


def remove_asfeature(solution):
    """
    Randomly excludes an as variable from solution generated from harmony consideration
    Inputs: solution list contianing all features 
    """
    if solution[0]:
        rem_asvar = list(np.random.choice(solution[0],1))
        solution[0] = [var for var in solution[0] if var not in rem_asvar]
        solution[0] = sorted(list(set(solution[0]).union(ps_asvars)))
        solution[2] = {k:v for k,v in solution[2].items() if k in solution[0]}
        solution[3] = [var for var in solution[3] if var in solution[0] and var not in ps_corvars]
        solution[4] = [var for var in solution[4] if var in solution[0] and var not in ps_bcvars]
    return(solution)


# In[42]:


def remove_isfeature(solution):
    """
    Randomly excludes an is variable from solution generated from harmony consideration
    Inputs: solution list contianing all features 
    """
    if solution[1]:
        rem_isvar = list(np.random.choice(solution[1],1))
        solution[1] = [var for var in solution[1] if var not in rem_isvar]
        solution[1] = sorted(list(set(solution[1]).union(ps_isvars)))
    return(solution)


# In[43]:


def remove_bcfeature(solution):
    """
    Randomly excludes a variable transformation from solution generated from harmony consideration
    Inputs: solution list contianing all features 
    """
    if solution[3]:
        rem_bcvar = list(np.random.choice(solution[3],1))
        rem_nps_bcvar = [var for var in rem_bcvar if var not in ps_bcvars]
        solution[3] = [var for var in solution[3] if var in solution[0] and var not in rem_nps_bcvar]
        solution[4] = [var for var in solution[4] if var not in solution[3]]
        solution[3] = [var for var in solution[3] if var not in solution[4]]
    return(solution)    


# In[44]:


def remove_corfeature(solution):
    """
    Randomly excludes correlaion feature from solution generated from harmony consideration
    Inputs: solution list contianing all features 
    """
    if solution[4]:
        rem_corvar = list(np.random.choice(solution[4],1))
        rem_nps_corvar = [var for var in rem_corvar if var not in ps_corvars]
        solution[4] = [var for var in solution[4] if var in solution[2].keys() and var not in rem_nps_corvar]
        if len(solution[4]) < 2:
            solution[4] = [] 
    return(solution)


# def assess_sol(solution,har_mem,seed):
#     """
#     (1) Evaluates the objective function of a given solution
#     (2) Evaluates if the solution provides an improvement in BIC by atleast a threshold value compared to any other solution in memory
#     (3) Checks if the solution is unique to other solutions in memory
#     (4) Replaces the worst solution in memory, if (2) and (3) are true
#     Inputs: solution list contianing all features, harmony memory
#     """
#     data = df.copy()
#     improved_sol,conv = evaluate_objective_function(data,seed,solution[0],
#                                                   solution[1],solution[2],solution[3],solution[4],choice_var,alt_var,choice_id,ind_id,solution[5])
#     if conv:
#         if all(har_mem[sol][0] != improved_sol[0] for sol in range(len(har_mem))):
#             if har_mem[-1][0]-improved_sol[0] >= threshold:
#                 if all(abs(har_mem[sol][0]-improved_sol[0]) >= threshold for sol in range(len(har_mem))):
#                     har_mem[-1] = improved_sol
#     har_mem = sorted(har_mem, key = lambda x: x[0])
#     return(har_mem,improved_sol)

# In[45]:


def assess_sol(solution,har_mem,seed):
    """
    (1) Evaluates the objective function of a given solution
    (2) Evaluates if the solution provides an improvement in BIC by atleast a threshold value compared to any other solution in memory
    (3) Checks if the solution is unique to other solutions in memory
    (4) Replaces the worst solution in memory, if (2) and (3) are true
    Inputs: solution list contianing all features, harmony memory
    """
    data = df.copy()
    improved_sol,conv = evaluate_objective_function(data,seed,solution[0],
                                                  solution[1],solution[2],solution[3],solution[4],choice_var,alt_var,choice_id,ind_id,solution[5])
    if conv:
        if all(har_mem[sol][0] != improved_sol[0] for sol in range(len(har_mem))):
            if all(har_mem[sol][0] - improved_sol[0] >= threshold for sol in range(1,len(har_mem))):
                if all(abs(har_mem[sol][0]-improved_sol[0]) >= threshold for sol in range(len(har_mem))):
                    har_mem[-1] = improved_sol
    har_mem = sorted(har_mem, key = lambda x: x[0])
    return(har_mem,improved_sol)


# In[46]:


def pitch_adjustment(sol, har_mem, PAR_itr,seeds,itr):
    seed = seeds[HMS*3+itr]
    """
    (1) A random binary indicator is generated. If the number is 1, then a new feature is added to the solution 
    generated in the Harmony Memory consideration step. Else a feature is randomly excluded from the solution
    (2) The objective function of a given solution is evaluated.
    (3) The worst solution in harmony memory is repalced with the solution, if it is unique and provides an improved BIC
    
    Inputs:
    solution list generated from harmony consideration step
    harmony memory
    Pitch adjustment rate for the given iteration
    """
    improved_harmony = har_mem
    if  np.random.choice([0,1], p=[1-PAR_itr,PAR_itr]) <= PAR_itr:
        if np.random.randint(2):
            print("pitch adjustment adding as variables")
            pa_sol = add_new_asfeature(sol,seed)
            improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if isvarnames:
                print("pitch adjustment adding is variables")
                pa_sol = add_new_isfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if ps_bctrans == None or ps_bctrans == True:
                print("pitch adjustment adding bc variables")
                pa_sol = add_new_bcfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if ps_cor == None or ps_cor == True:
                print("pitch adjustment adding cor variables")
                pa_sol = add_new_corfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
        
        elif len(sol[0])>1:
            print("pitch adjustment by removing as variables")
            pa_sol = remove_asfeature(sol)
            improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if isvarnames or sol[1]:
                print("pitch adjustment by removing is variables")
                pa_sol = remove_isfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if ps_bctrans == None or ps_bctrans == True:
                print("pitch adjustment by removing bc variables")
                pa_sol = remove_bcfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
            
            if ps_cor == None or ps_cor == True:
                print("pitch adjustment by removing cor variables")
                pa_sol = remove_corfeature(pa_sol)
                improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
        else:
            print("pitch adjustment by adding asfeature")
            pa_sol = add_new_asfeature(sol,seed)
            improved_harmony, current_sol = assess_sol(pa_sol,har_mem,seed)
    else:
        print("no pitch adjustment")
        improved_harmony, current_sol = assess_sol(sol,har_mem,seed)
    return(improved_harmony, current_sol)


# ## 7.c Local Search

# In[47]:


def best_features(har_mem):
    """
    Generates lists of best features in harmony memory
    Inputs:
    Harmony memory
    """
    best_asvars = har_mem[0][1].copy()
    best_isvars = har_mem[0][2].copy()
    best_randvars = har_mem[0][3].copy()
    best_bcvars = har_mem[0][4].copy()
    best_corvars = har_mem[0][5].copy()
    asc_ind = har_mem[0][6]
    return(best_asvars,best_isvars,best_randvars,best_bcvars,best_corvars,asc_ind)


# In[48]:


from matplotlib.backends.backend_pdf import PdfPages
def local_search(improved_harmony,seeds,itr): 
    seed = seeds[HMS*4+itr]
    """
    Initiate Artificial Bee-colony optimization
    ##Check if tweeking the best solution in harmony improves solution's BIC
    Inputs: improved memory after harmony consideration and pitch adjustment
    """
    #For plots (BIC vs. iterations)
    best_bic_points = []
    current_bic_points = []
    x=[]
    #pp = PdfPages('BIC_plots_localsearch.pdf') 
    
    #Select best solution features
    best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
    
    print("first set of best features input for local search",best_asvars,best_isvars,best_randvars,best_bcvars,best_corvars)
    #for each additional feature to the best solution, the objective function is tested
    
    #if all variables in varnames are present in the best solution, we improvise the solution by changing some features
    if len(best_asvars) == len(asvarnames):
        ##Check if changing coefficient distributions of best solution improves the solution BIC
        for var in best_randvars.keys():
            if var not in ps_rvars:
                rm_dist = [dis for dis in dist if dis != best_randvars[var]]
                best_randvars[var] = np.random.choice(rm_dist)
        best_randvars = {key:val for key,val in best_randvars.items() if key in best_asvars and val != 'f'}
        best_bcvars = [var for var in best_bcvars if var in best_asvars and var not in ps_corvars]
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
        solution_1 = [best_asvars, best_isvars,best_randvars,best_bcvars,best_corvars,asc_ind]
        improved_harmony, current_sol = assess_sol(solution_1,improved_harmony,seed)
        print("sol after local search step 1", improved_harmony[0])
                    
        ##check if having a full covariance matrix has an improvement in BIC
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
        best_bcvars = [var for var in best_asvars if var in ps_bcvars]
        if ps_cor == None or ps_cor == True:
            best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
        elif len(best_corvars)<2:
            best_corvars = []
        else:
            best_corvars = []
        solution_2 = [best_asvars, best_isvars,best_randvars,best_bcvars,best_corvars,asc_ind]
        improved_harmony, current_sol = assess_sol(solution_2,improved_harmony,seed)
        print("sol after local search step 2", improved_harmony[0])
        
        
        ##check if having a all the variables transformed has an improvement in BIC
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
        if ps_bctrans == None or ps_bctrans == True:
            best_bcvars = [var for var in best_asvars if var not in ps_corvars]
        else:
            best_bcvars = []
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
        solution_3 = [best_asvars, best_isvars,best_randvars,best_bcvars,best_corvars,asc_ind]
        improved_harmony, current_sol = assess_sol(solution_3,improved_harmony,seed)
        print("sol after local search step 3", improved_harmony[0])
                    
    else:
        print("local search by adding variables")
        solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
        solution_4 = add_new_asfeature(solution,seed)
        improved_harmony, current_sol = assess_sol(solution_4,improved_harmony,seed)
        print("sol after local search step 4", improved_harmony[0])
        
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
        solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
        solution_5 = add_new_isfeature(solution)
        improved_harmony, current_sol = assess_sol(solution_5,improved_harmony,seed)
        print("sol after local search step 5", improved_harmony[0])
        
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
        solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
        solution_6 = add_new_bcfeature(solution)
        improved_harmony, current_sol = assess_sol(solution_6,improved_harmony,seed)
        print("sol after local search step 6", improved_harmony[0])
        
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
        solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
        solution_7 = add_new_corfeature(solution)
        improved_harmony, current_sol = assess_sol(solution_7,improved_harmony,seed)
        print("sol after local search step 7", improved_harmony[0])
        
        
    ## Sort unique harmony memory from min.BIC to max. BIC
    improved_harmony = sorted(improved_harmony, key = lambda x: x[0])

    ##Check if changing coefficient distributions of best solution improves the solution BIC
    best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
    
    for var in best_randvars.keys():
        if var not in ps_rvars:
            rm_dist = [dis for dis in dist if dis != best_randvars[var]]
            best_randvars[var] = np.random.choice(rm_dist)
    best_randvars = {key:val for key, val in best_randvars.items() if key in best_asvars and val != 'f'}
    best_bcvars = [var for var in best_bcvars if var in best_asvars and var not in ps_corvars]
    if ps_cor == None or ps_cor == True:
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
    elif ps_cor == False:
        best_corvars = []
    if len(best_corvars)<2:
        best_corvars = []
    solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
    improved_harmony, current_sol = assess_sol(solution,improved_harmony,seed)
    print("sol after local search step 8", improved_harmony[0])
    
    
    ##check if having a full covariance matrix has an improvement in BIC
    best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
    best_bcvars = [var for var in best_asvars if var in ps_bcvars]
    if ps_cor == None or ps_cor == True:
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
    else:
        best_corvars = []
    if len(best_corvars)<2:
        best_corvars = []
    solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
    improved_harmony, current_sol = assess_sol(solution,improved_harmony,seed)
    print("sol after local search step 9", improved_harmony[0])
   


    ##check if having all the variables transformed has an improvement in BIC
    best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind = best_features(improved_harmony)
    if ps_bctrans == None or ps_bctrans == True:
        best_bcvars = [var for var in best_asvars if var not in ps_corvars]
    else:
        best_bcvars = []
    if ps_cor==None or ps_cor==True:
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
    else:
        best_corvars = []
    if len(best_corvars)<2:
        best_corvars = []
    solution = [best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind]
    improved_harmony, current_sol = assess_sol(solution,improved_harmony,seed)
    print("sol after local search step 10", improved_harmony[0])
    
                    
    ## Sort unique harmony memory from min.BIC to max. BIC
    final_harmony_sorted = sorted(improved_harmony, key = lambda x: x[0])
    return(final_harmony_sorted,current_sol)


# ## 7.d Improve Harmony

# In[49]:


##Function to conduct harmony memory consideraion, pitch adjustment and local search

def improvise_harmony(HCR_max,HCR_min,PR_max,PR_min,har_mem,max_itr,threshold,itr_prop):
    improve_harmony =  code_name + 'improvise_harmony_' + current_date + '.txt'
    sys.stdout = open(improve_harmony,'wt')
    itr = 0
    
    #for BIC vs. iteration plots
    best_bic_points = []
    current_bic_points = []
    x = []
    pdf_name = 'BIC_plots_final_' + code_name + current_date + '.pdf'
    pp = PdfPages(pdf_name)
    #np.random.seed(500)
    np.random.seed(seeds[itr])
    while itr < max_itr:
        itr+= 1
        #print(itr)
        #Estimate dynamic HMCR and PCR values for each iteration
        HMCR_itr = (HCR_min + ((HCR_max-HCR_min)/max_itr)*itr) * max(0,math.sin(itr))
        PAR_itr = (PR_min + ((PR_max-PR_min)/max_itr)*itr) * max(0,math.sin(itr)) 
        #seed = seeds[itr]
        #Conduct Harmony Memory Consideration
        hmc_sol = harmony_consideration(har_mem, HMCR_itr,seeds,itr)
        print("solution after HMC at iteration",itr, "is", hmc_sol)
        #Conduct Pitch Adjustment
        pa_hm, current_sol = pitch_adjustment(hmc_sol,har_mem,PAR_itr,seeds,itr)
        print("best solution after HMC & PA at iteration",itr, "is", pa_hm[0])
        current_bic_points.append(current_sol[0])
        ## Sort unique harmony memory from min.BIC to max. BIC
        har_mem_sorted = sorted(pa_hm, key = lambda x: x[0])
        ## Trim the Harmony memory's size as per the harmony memory size
        har_mem = har_mem_sorted[:HMS]
        #Append y-axis points for the plots
        best_bic_points.append(har_mem[0][0])
        x.append(itr)
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('BIC')
        #plt.plot(x, current_bic_points, label = "BIC from current iteration")
        plt.plot(x, best_bic_points, label = "BIC of best solution in memory")
        #pp.savefig(plt.gcf())
        plt.show()
        sys.stdout.flush()
        
        #check iteration to initiate local search
        if itr > int(itr_prop*max_itr):
            print("HM before starting local search",har_mem)  
            print("local search initiated at iteration", itr)
            #seed = seeds[itr]
            har_mem, current_sol = local_search(har_mem,seeds,itr)
            ## Sort unique harmony memory from min.BIC to max. BIC
            har_mem = sorted(har_mem, key = lambda x: x[0])
            ## Trim the Harmony memory's size as per the harmony memory size
            har_mem = har_mem[:HMS]
            
            print("final harmony in current iteration", itr, "is", har_mem)
            
            best_bic_points.append(har_mem[0][0])
            current_bic_points.append(current_sol[0])
            print(har_mem[0][0])
            x.append(itr)
            plt.plot(current_bic_points)
            plt.plot(best_bic_points, linestyle='--')
            plt.legend(["new_harmony", "best_harmony"])
            plt.show()
            """
            #plt.figure()
            #plt.xlabel('Iterations')
            #plt.ylabel('BIC')
            #plt.plot(x, current_bic_points, label = "BIC from current iteration")
            plt.plot(x, best_bic_points, label = "BIC of best solution in memory")
            #pp.savefig(plt.gcf())
            plt.show() 
            """
            sys.stdout.flush()
            if itr == max_itr+1:
                break
    pp.close()
    
    sys.stdout.flush()
    return(har_mem,best_bic_points,current_bic_points)


# In[50]:


Initial_harmony = hm.copy()
new_HM, best_BICs, current_BICs = improvise_harmony(HMCR_max,HMCR_min,PAR_max,PAR_min,Initial_harmony,itr_max,threshold,v)
improved_harmony = new_HM.copy()


# In[51]:


benchmark_bic = improved_harmony[0][0]
best_asvarnames = improved_harmony[0][1]
best_isvarnames = improved_harmony[0][2]
best_randvars = improved_harmony[0][3]
best_bcvars = improved_harmony[0][4]
best_corvars = improved_harmony[0][5]
best_Intercept = improved_harmony[0][6]
benchmark_bic,best_asvarnames,best_isvarnames,best_randvars,best_bcvars,best_corvars,best_Intercept


# In[52]:


print("Search ended at",time.ctime())


# # Final best model

# best_asvarnames = ['risk', 'convloc', 'noise', 'seats', 'clientele', 'cost', 'crowdness']
# best_isvarnames = ['african']
# best_randvars = {'convloc': 'ln', 'noise': 'tn', 'clientele': 'ln', 'crowdness': 't'}
# best_bcvars = []
# best_corvars = []
# best_Intercept = False

# In[53]:


best_varnames = best_asvarnames + best_isvarnames
if bool(best_randvars):
    model = MixedLogit()
    model.fit(X=df[best_varnames], y=choice_var,varnames=best_varnames, isvars = best_isvarnames, alts=alt_var, ids=choice_id, panels=ind_id, randvars=best_randvars, transformation= "boxcox",transvars=best_bcvars,correlation=best_corvars,fit_intercept=best_Intercept, n_draws=200)
else:
    model = MultinomialLogit()
    model.fit(X=df[best_varnames], y=choice_var,varnames=best_varnames,isvars = best_isvarnames, alts=alt_var, ids=choice_id, transformation= "boxcox",transvars=best_bcvars,fit_intercept=best_Intercept)
print(model.summary())


# In[54]:


print(best_BICs)


# In[55]:


print(current_BICs)


# In[56]:


sys.stdout.flush()


# In[ ]:




