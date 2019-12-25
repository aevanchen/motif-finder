import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import log1p
import math
from random import random
from scipy.stats import norm
from tqdm import tqdm,tqdm_notebook
from scipy.stats.distributions import chi2
import difflib
from utlis import *
from motif_class import *
from display import *
import pandas as pd
from collections import defaultdict
elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}


    
def MEME( model_type):
    filename="test3.fa"
    #filename="test-motif.fa"
    seqs=read_fasta(filename)
    
   
    beta=0.5# the Dirichlet prior,

    epsilon=0.8
    p=get_p(beta)
    npass=2

    Wmin=4

    max_motif_length= int(0.8* min([len(seq) for seq in seqs]))
    
    Wmax=min(8,max_motif_length) 

    U= [[1  for i in range(len(seq))]for seq in seqs]

    theta_0_from_test,cnt_dict=init_theta_0(seqs)
    #null_theta_0=log(np.array([0.25,0.25,0.25,0.25]))
    model_type=model_type
    test_dict={}
    
    Wlist=range(Wmin,Wmax+1)
    #Wlist=[18,19,30,45]
    for W in Wlist:
            lambda_list=get_lambda_list(seqs,W,model_type=model_type)
            max_theta1_from_test,max_z=test(seqs,W,lambda_list,p,theta_0_from_test,epsilon,quick_test=True,model_type=model_type)

            test_dict[W]=[max_theta1_from_test,lambda_list,max_z]

    
    for ipass in range(npass):
        min_G=float('inf')
        target_consensus_motif=None
        target_W=None
        target_z=None
        target_theta=None
        target_z_expected=None
        tartget_motifs=None
      
        
        for W in Wlist:
                print('W=', W)
                lambda_list=test_dict[W][1]
    
                
                for lambda_i,lambda_value in enumerate(lambda_list):
                
                    theta_1_from_test=test_dict[W][0][lambda_i] 
                    
                    
                    likelihood,z_list,theta_1,theta_0,_=EM(seqs,W,theta_0_from_test,theta_1_from_test,U,epsilon,lambda_value,model_type)  
                    
                    null_likelihood=get_null(cnt_dict)
                   # null_likelihood=get_null_model_likelihood(seqs, theta_0_from_test)  
                    
                    G=compute_G(likelihood,null_likelihood,W,ipass,model_type=model_type)
                    
                    #print(likelihood,null_likelihood)
#                     print(consensus_motif)
                    #print(likelihood,null_likelihood,G)
                    if G<min_G:
                        
                        min_G=G
                        print(min_G)
#                         target_consensus_motif=consensus_motif
                        target_W=W
                        target_z_expected=z_list
                        pad= max([ len(i) for i in z_list])
        
                        pr_list_padded=np.array([i + [-float('inf')]*(pad-len(i)) for i in z_list])

                        target_z=get_target_z(seqs,pr_list_padded,W,model_type=model_type)
                        tartget_motifs=get_motif_seqs(seqs,target_z,W)
                        print(target_z)
                        print( tartget_motifs)
                        target_consensus_motif=majority(tartget_motifs)
                        target_theta=theta_1_from_test
                     
          
        #update U
        
        cnt_dict=update_dict(cnt_dict, tartget_motifs)
        U=update_U(U,target_z_expected,target_W) 
        theta_0_from_test=cnt_dict/np.sum(cnt_dict)
        
        if target_consensus_motif=="":
            print("No motif found at pass {}".format(ipass+1))
        else:
            print("The consenus motif is found at pass {} is {}".format(ipass+1,target_consensus_motif))
        print(cnt_dict,theta_0_from_test)
        
        #print(U)
        print(min_G)
        print(target_W)
        print(target_z)
#         print( [np.argwhere(i)[0] for i in target_z])
        print(tartget_motifs)

       
    return 

       
def test_EM(seqs,W,theta_0_from_test,theta_1_from_test,U,epsilon,lambda_value,model_type):
    
    theta_0=theta_0_from_test
    theta_1=theta_1_from_test
    max_theta1=None
    max_likelihood=-float('inf')
    max_z=None
    max_subsequence=None
    nc=compute_nc(seqs)
    prev_theta=None
    prev_likelihood=None
    threshold=0.00001
    t=0
    
    V=get_pr_V(U,W)
    
    likelihood_diff=[]
    theta_diff=[]
    likelihood_dict=[]
    theta_dict=[]
    for kk in range(100):
        t+=1
        # E step
        
        z_expected=Estep(seqs,W,theta_0,theta_1,lambda_value, model_type)         
      
        #add v constraints
        z_expected=[[z_expected[i][j]*V[i][j] for j in range(len(V[i]))] for i in range(len(V))]
      
        # M step 
      
        theta_0,theta_1=compute_theta(seqs,W,z_expected,nc,kk,epsilon=epsilon)
   
        
        if model_type=='ZOOPS' or model_type=='TCM':    
            lambda_value= update_lambda(z_expected)
 
        
        likelihood=compute_joint_likelihood(seqs,W,theta_0,theta_1,z_expected,lambda_value,model_type=model_type)
        if prev_likelihood is not None:
            likelihood_dict.append(likelihood)
            
            likelihood_diff.append(likelihood-prev_likelihood)
        if prev_theta is not None:
            theta_dict.append(np.sum(theta_1))
            
            theta_diff.append(get_theta_different(prev_theta,theta_1))

   #         if  get_theta_different(prev,theta_1)<threshold:
#             break
#         if prev is not None and likelihood>prev and likelihood-prev<0.0001:
#             print(kk,likelihood-prev)
#             break
        prev_likelihood=likelihood
        prev_theta=theta_1
        
    
    return  likelihood_dict,theta_dict,likelihood_diff, theta_diff



filename="crp.fa"
#filename="test-motif.fa"
seqs=read_fasta(filename)

model_type='OOPS'
beta=0.5# the Dirichlet prior,

epsilon=0.25
p=get_p(beta)
npass=2

Wmin=4

max_motif_length= int(0.8* min([len(seq) for seq in seqs]))

Wmax=min(8,max_motif_length) 

U= [[1  for i in range(len(seq))]for seq in seqs]

theta_0_from_test,cnt_dict=init_theta_0(seqs)
#null_theta_0=log(np.array([0.25,0.25,0.25,0.25]))
model_type=model_type
test_dict={}

Wlist=range(Wmin,Wmax+1)
#Wlist=[18,19,30,45]
W=20
lambda_list=get_lambda_list(seqs,W,model_type=model_type)
max_theta1_from_test,max_z=test(seqs,W,lambda_list,p,theta_0_from_test,epsilon,quick_test=True,model_type=model_type)

test_dict[W]=[max_theta1_from_test,lambda_list,max_z]
min_G=float('inf')
target_consensus_motif=None
target_W=None
target_z=None
target_theta=None
target_z_expected=None
tartget_motifs=None
      

print('W=', W)
lambda_list=test_dict[W][1]


for lambda_i,lambda_value in enumerate(lambda_list):

    theta_1_from_test=test_dict[W][0][lambda_i] 


    
    likelihood_dict,theta_dict,likelihood_diff, theta_diff=test_EM(seqs,W,theta_0_from_test,theta_1_from_test,U,epsilon,lambda_value,model_type)  

    break
    
    

plt.figure()
x=range(1,len(likelihood_dict)+1)

plt.plot(x, likelihood_dict, label ="Moving Sum in joint loglikelihood")
plt.legend()
plt.title("The statility of convergence using likelihood criterion")
plt.xticks(np.arange(0, len(likelihood_dict)+1,10))
plt.xlim([0,30])
plt.xlabel("EM Iters")
plt.ylabel("Value")
plt.savefig("value_W=20_ll.jpg")



plt.figure()
x=range(1,len(theta_dict)+1)

plt.plot(x, theta_dict, label ="Moving sum in PWM matrxi")
plt.legend()
plt.title("The statility of convergence using PWM criterion")
plt.xticks(np.arange(0, len(theta_dict)+1,10))
plt.xlabel("EM Iters")
plt.xlim([0,30])
plt.ylabel("Value")
plt.savefig("value_W=20_pwm.jpg")



markers = ['^', 'o', 'x', '<', 's', '+', 'p', '>' ]
plt.figure()
x=range(1,len(likelihood_diff)+1)

plt.plot(x, likelihood_diff, label ="Moving diff in joint loglikelihood")
plt.legend()
plt.title("The statility of convergence using likelihood criterion")
plt.xticks(np.arange(0, len(likelihood_diff)+1,10))
plt.xlabel("EM Iters")
plt.yscale("log")
plt.xlim([0,30])
plt.ylabel("Value Difference in log")
plt.savefig("value_diff_W=20_ll.jpg")


markers = ['^', 'o', 'x', '<', 's', '+', 'p', '>' ]
plt.figure()
x=range(1,len(theta_diff)+1)

plt.plot(x, theta_diff, label ="Moving diff in PWM matrix")
plt.legend()
plt.title("The statility of convergence using PWM criterion")
plt.xticks(np.arange(0, len(theta_diff)+1,10))
plt.xlabel("EM Iters")
plt.yscale("log")
plt.xlim([0,30])
plt.ylabel("Value Difference in log")
plt.savefig("value_diff_W=20_pwm.jpg")