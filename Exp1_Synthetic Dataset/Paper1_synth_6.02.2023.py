#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install searchlogit')


# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install searchlogit -U


# In[2]:


import pandas as pd
import numpy as np
from searchlogit import device as dev
from scipy.special import boxcox, inv_boxcox
print(dev.get_device_count())


# In[ ]:


#from searchlogit import MixedLogit
#MixedLogit.check_if_gpu_available()


# In[3]:


# In[3]:


#df = pd.read_csv("artificial_corr_other_distributions.csv") #"("training.csv")
df = pd.read_csv("artificial_1h_mixed_corr_trans_new_final.csv") #"("training.csv") # panel dataset


# In[4]:


### Include manually transformed variables
df['bc_added_random4'] = boxcox(df['added_random4'], 0.4)
df['bc_added_random5'] = boxcox(df['added_random5'], 0.3)


# In[5]:


df.columns


# In[ ]:


from searchlogit import Search
choice_id =df['choice_id']  # TODO: CHANGED
#test_chid = df_test['id']

ind_id = df['ind_id']
#test_ind_id = df_test['ind']

alt_var = df['alt'] # TODO: CHANGED
#test_alt_var = df_test['alt'] 

av = None
#test_av = None# df_test['AV']

weight_var = None 
#test_weight_var = None 

dist = ['n', 'f', 'u', 't']
choice_set=['1', '2', '3']

asvarnames = ['added_fixed1', 'added_fixed2',
       'added_fixed3', 'added_fixed4', 'added_fixed5', 'added_fixed6',
       'nonsig1', 'nonsig2', 'nonsig3', 'nonsig4', 'nonsig5', 'added_random1',
       'added_random2', 'added_random3', 'added_random4', 'added_random5',
       'added_random6', 'added_random7', 'bc_added_random4',
       'bc_added_random5']
isvarnames = []
varnames = asvarnames + isvarnames
trans_asvars = []

choice_var = df['choice']
#test_choice_var = df_test['choice'] # CHANGED the df column name containing the choice variable

X = df[varnames].values
y = df['choice'].values

base = None

#intercept_opts = {'class_intercept_alts': [[2,4,5,6], [2,3,4,5,6]]}

#avail_N = len(np.unique(df['custom_id']))
#avail_latent = [np.tile([1, 1, 0, 1, 1, 1], ((avail_N, 1))),
                #None]

search = Search(df=df, 
                choice_set=choice_set, 
                alt_var=alt_var,
                #test_alt_var=test_alt_var,
                varnames=varnames,isvarnames=isvarnames,
                asvarnames=asvarnames,
                choice_var=choice_var,
                #test_choice_var=test_choice_var,
                choice_id=choice_id,
                #test_choice_id= test_chid,
                ind_id=ind_id, 
                #test_ind_id=test_ind_id,
                latent_class=False,
                multi_objective=False,
                #gtol=1e-2,
                allow_random=True,
                base_alt = base,
                allow_bcvars=False,
                # intercept_opts=intercept_opts,
                # avail_latent=avail_latent,
                seed=1,
                n_draws=1000,
                code_name="Synth_SOOF",
               dist = dist)
#search.ps_asvars = ['TT_CAD', 'TT_CAP', 'TT_W2PT', 'TT_KR', 'TT_PR', 'TT_WALK', 'TT_CYCLE','TCPC', 'TC_W2PT', 'TC_KR', 'TC_PR']
HMS = search.run_search(HMS=5, itr_max=500)


# In[ ]:


"""
from searchlogit import MixedLogit
model = MixedLogit()
model.fit(X=df[base_varnames].values, y=choice_var, varnames=base_varnames, isvars=base_isvarnames, alts=alt_var,
          randvars=baservars,n_draws= R, transvars=basebcvars,correlation=basecorvars, ids=choice_id, avail=av,fit_intercept=base_intercept, base_alt = base,gtol=1e-02,ftol=1e-02)
model.summary()
conv = model.convergence
"""


# In[ ]:


"""
choice_id = df['id']
test_chid = df_test['id']
ind_id = df['id']
test_ind_id = df_test['id']
base_varnames = ['added_fixed3', 'added_random3', 'conven', 'emipp', 'meals', 
          'rand_nonlinear1', 'rand_nonlinear2']
base_isvarnames = []
baservars = {}
basebcvars = []
basecorvars = []
choice_set=['1','2','3']
choice_var = df['choice']
test_choice_var = df_test['choice']
alt_var = df['alt']
test_alt_var = df_test['alt']
Tol = 1e-02
#FTol = 1e-02
base_intercept = False #True
#dist = ['n', 'ln', 'tn', 'u', 't', 'f']
av = None
test_av = None
weight_var = None
test_weight_var = None
base = None
#dist = ['n', 'tn', 'u', 'f']
val_share = 0.25 #Proportion of sample held out for validation
R = 1000


from searchlogit import MultinomialLogit
model = MultinomialLogit()
model.fit(X=df[base_varnames].values, y=choice_var, varnames=base_varnames, isvars=base_isvarnames, alts=alt_var,
          ids=choice_id, avail=av,fit_intercept=base_intercept, base_alt = base,gtol=1e-02,ftol=1e-02)
model.summary()
conv = model.convergence
"""


# In[ ]:




# In[ ]:




