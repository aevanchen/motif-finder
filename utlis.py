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
elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}

def log(x):
    x=np.array(x)
    x[x<0]=0
    return  np.log(x+1e-200)

def read_fasta(filename):
    with open(filename, "r") as f:
        output = []
        s = ""
        for l in f.readlines():
            if l.strip()[0] == ">":
                # skip the line that begins with ">"
                if s == "": continue
                output.append(s)
                s = ""
            # keep appending to the current sequence when the line doesn't begin
            # with ">"
            else:
                s += l.strip().upper()
        output.append(s)
        return output


def sumLogProb(a, b):
    if a > b:
        return a + np.log1p(math.exp(b - a))
    else:
        return b + np.log1p(math.exp(a - b))


def sumlogProbList(l):
    # likelihood = logsumexp([hl, hh, lh, lh])
    likelihood = -math.inf
    for i in l:
        likelihood = sumLogProb(likelihood, i)

    return likelihood


def init_theta_0(seqs):
    theta_0=np.zeros(len(dict_map.keys()))
    total=sum( [len(seq) for seq in seqs])
    for seq in seqs:
        for ele in seq:
            theta_0[dict_map[ele]]+=1
    return log(theta_0/total)

def compute_nc(seqs):
    nc=np.zeros(len(dict_map.keys()))
    for seq in seqs:
        for ele in seq:
            nc[dict_map[ele]]+=1
    return nc

# Pr( Xij/theta1) 
# seq: the ith seq
# j: subsequence start index
def compute_pr_theta_1(seqi, j,W,theta1):
    subsequnce=seqi[j:j+W]
   
    res=0
    for i,ele in enumerate(subsequnce):
    
        index=dict_map[ele]
        res+=theta1[index,i]
    
    
    return res


def compute_pr_theta_0(seqi, j,W,theta0):
    subsequnce=seqi[j:j+W]
   
    res=0
    for i,ele in enumerate(subsequnce):
        index=dict_map[ele]
        res+=theta0[index]
    
    
    return res
# get pr/given theta
def get_pr_given_theta(sequence, s, theta_0,theta_1,W):

    t = len(sequence)
    res = 0
    for i in range(t):
        if i>=s and i<=s+W-1:
            res += theta_1[ dict_map[sequence[i]],i - s]
        else:
            res += theta_0[dict_map[sequence[i]]]
    return res


# Pr( Xij/theta1) using dp
# pr: previous pr_theta_1

# def compute_pr_theta_1_fast(pr,seqi, j,W,theta1):
#     start=dict_map[seqi[j-1]]
#     end=dict_map[seqi[j+W-1]]

#     res=pr-theta1[start,0]+theta1[end,W-1]
#     return res

def compute_pr_theta_1_dp(seqs,prev_pr_list,current_p,prev_p,i,j,W):
    
    minus=prev_p[dict_map[seqs[i][j]],0]
    add=current_p[dict_map[seqs[i][j+W]],W-1]
    return prev_pr_list[i][j]-minus+add
    
#set p: a vector [pa,pc,pg,pt]
# give beta, the Dirichlet prior, dna=0.52, proteain=0.15 
# the remain entry is (1-pi)/len(elements)-1
def get_p(beta):
    m=len(elements)
    p=np.full((m, m), beta)
    np.fill_diagonal(p, 4)
    p=p/np.sum(p,axis=1)
    return p


def init_theta_1(subsequence,p):
    # assert sum([s+k<len(sequences[i]) for i,s in enumerate(starts)])==0
   # inverse_dict_map = {i: ele for i, ele in enumerate(['A', 'C', 'G', 'T'])}
    index=[dict_map[ele] for ele in subsequence]
    #p[index]
    return  log(p[:,index])



def get_theta_1_in_test(seqs,sequence_ids,start_list,W,epsilon=1):
     
    #motifs= np.array([list(seqs[i][start:start+W])for i,start in enumerate(start_list) if i !=k])
    motifs= np.array([list(seqs[seq_id][start_list[i]:start_list[i]+W])for i,seq_id in enumerate(sequence_ids)])

    theta_1 = np.zeros((len(elements), W))
    for motif_pos in range(W):
        for i in range(len(elements)):
            
            theta_1[i,motif_pos] = np.sum(motifs[:,motif_pos] == elements[i])
       
        theta_1[ :,motif_pos] = (theta_1[:,motif_pos] + epsilon) / (len(motifs)+ 4 * epsilon)
    return log(theta_1)


def get_pr_given_q(seq,theta_0):
    t=len(seq)
    res=0
    for i in range(t):
         res += theta_0[dict_map[seq[i]]]
    return res


def compute_likelihood(seqs,W,theta_0,theta_1,sequence_ids,start_list,lambda_dict, model_type='OOPS'):
    res=0

    if model_type=='OOPS':
        
        for i,seq in enumerate(seqs):
            res+=log(lambda_dict[i])
            res+=get_pr_given_theta(seq,start_list[i], theta_0,theta_1,W)
    elif model_type=='ZOOPS':
        Q=np.zeros(len(seqs))
        Q[sequence_ids]=1
                
        for i,seq_id in enumerate(sequence_ids):
            if Q[seq_id]==1:
                res+=get_pr_given_theta(seqs[seq_id],start_list[i], theta_0,theta_1,W)
                res+=log(lambda_dict[seq_id])
                
            else:
                res+=get_pr_given_q(seqs[seq_id],theta_0)
                res+=log(1-(len(seqs[seq_id])-W+1)*lambda_dict[seq_id])
    else:

        for i,seq_id in enumerate(sequence_ids):
            
            res+=compute_pr_theta_1(seqs[seq_id],start_list[i],W,theta_1)
         
            
            res+=log(lambda_dict[seq_id])
            
            res-=log(1-lambda_dict[seq_id])
            res-=compute_pr_theta_0(seqs[seq_id],start_list[i],W,theta_0)

        for i, seq in enumerate(seqs):
            for j in range(len(seq)-W+1):
                res+=compute_pr_theta_0(seq, j,W,theta_0)
                res+=log(1-lambda_dict[i])
    return res
    
    
    

def compute_theta(seqs,W,z,nc,kk,epsilon=0.25):

    nck=[[0,0,0,0] for i in range(W+1)]
    t=[0,0,0,0]
    for k in range(1,W+1):
         for i,seq in enumerate(seqs):
            for j in range(0,len(seq)-W+1):

                indice=dict_map[seq[j+k-1]]
                nck[k][indice]+=z[i][j]           
                
    nck[0]=list(nc-np.sum(nck,axis=0))
    for i in range(len(nck[0])):
        nck[0][i]=max(0,nck[0][i])
    
    nck=np.array(nck)
    nck+=epsilon
    #nc=nc+epsilon
    theta=nck/np.sum(nck,axis=1).reshape(-1,1)

    theta_0=log(theta[0])
    theta_1=log(theta[1:].T)

    return theta_0,theta_1

def majority(seqs,z_list,W):
    
    
    start_list=np.array([np.argmax(i) for i in z_list])
    
    motif_sequences= np.array([list(seqs[i][s:s+W]) for i,s in enumerate(start_list)])
    
    from scipy import stats
    res=stats.mode(np.array(motif_sequences))[0]

    return res


def get_null_model_likelihood(seqs,theta_0=None):
    
    if theta_0 is None:
        theta_0=init_theta_0(seqs)
    null_likelihood=0
    total=0
    for seq in seqs:
        total+=len(seq)     
    for i in range(len(theta_0)):
        null_likelihood+=np.exp(theta_0[i])*theta_0[i]
        
   #null_likelihood=(total-target_W*len(seqs))*null_likelihood
    null_likelihood=total*null_likelihood
    return null_likelihood

def compute_G(log_likelihood,null_loglikelihood,W, n_pass,model_type='OOPS',palindrome=False,):  
    # in the second pass we use the joint loglikelihood  of the entire sequneces as criterion
    #if n_pass>=1:
       # return -log_likelihood
    
    if null_loglikelihood>=log_likelihood and model_type!='TCM':
      
        return 1
    elif null_loglikelihood>=log_likelihood and model_type=='TCM':
        null_loglikelihood=null_loglikelihood*10
        if null_loglikelihood>=log_likelihood :
            return 1
        
    chi=2*(log_likelihood-null_loglikelihood)
    #apply palindrome constraints
    if palindrome:
        v=W/2*3
    else:
        v=W*3
    x2=(((chi/v)**(1/3)-(1-2/9/v))/np.sqrt(2/(9*v)))
  
    lrt=norm.logsf(x2)
 
    G=(1/v)*lrt

    return G

def get_indicator_z(seqs,pos):
    z_list= [[0 for i in range(len(seq))] for seq in seqs]

    n_pos=len(pos[0])
    for i in range(n_pos):
        seq_id=pos[0][i]
        start_id=pos[1][i]
        z_list[seq_id][start_id]=1
    return z_list


def majority(motifs):
    motif_flatten=[]
    for motif in motifs:
        motif_flatten+=motif
   
    motif_sequences=np.array([list(s) for s in motif_flatten])

    from scipy import stats
    res=stats.mode(np.array(motif_sequences))[0][0]

    return "".join(res)

def get_motif_seqs(seqs,z_list,W):
    motifs=[]
    for i,z in enumerate(z_list):
        if sum(z)==0:
            temp=[]
            motifs.append(temp)
        else:
            temp=[]
            index_list=np.where(z)[0]
            for k in index_list:

                temp.append(seqs[i][k:k+W])
            motifs.append(temp)
    return motifs

def get_indicator_z(seqs,pos):
    z_list= [[0 for i in range(len(seq))] for seq in seqs]

    n_pos=len(pos[0])
    for i in range(n_pos):
        seq_id=pos[0][i]
        start_id=pos[1][i]
        z_list[seq_id][start_id]=1
    return z_list


def get_motif_seqs(seqs,z_list,W):
    motifs=[]
    for i,z in enumerate(z_list):
        if sum(z)==0:
            temp=[]
            motifs.append(temp)
        else:
            temp=[]
            index_list=np.where(z)[0]
            for k in index_list:

                temp.append(seqs[i][k:k+W])
            motifs.append(temp)
    return motifs

def prepare_pos(seqs,pr_list_padded,W,model_type='OOPS'):
    if model_type=='OOPS':
        pos=(np.arange(0,len(pr_list_padded)),np.argsort(-pr_list_padded, axis=1)[:,0])

    elif model_type=='ZOOPS':
        
        pos=(np.arange(0,len(pr_list_padded)),np.argsort(-pr_list_padded, axis=1)[:,0])
        
        pos_index=np.argsort(-pr_list_padded[pos])
        pos=tuple([i[pos_index] for i in pos ])
        
    
        

    else:
        for i,seq in enumerate(seqs):
                start=0
                for j in range(1,len(seq)-W+1):
                    if start+W+1==j:
                        start=j
                        continue         
                    
                    pr_list_padded[i][j]=-float('inf')

        pos=np.unravel_index(np.argsort(-pr_list_padded, axis=None), pr_list_padded.shape)
    return pos

def test(seqs,W,lambda_list,p,theta_0,epsilon=1,quick_test=True,model_type='OOPS'):
    
    max_theta1_from_test=[None]*len(lambda_list)
    max_likelihood=[-float('inf')]*len(lambda_list)
    max_z=[None]*len(lambda_list)
    max_subsequence=[None]*len(lambda_list)
    set_motif={}
    prev_p=[None]*len(lambda_list)
    prev_pr_list=[None]*len(lambda_list)
    n=len(seqs)
   
    for k,outer_seq in enumerate(seqs):
        
        if quick_test and k==1:
            break
        for l in range(len(outer_seq)-W+1):
            #map sequence Xkl to theta 1
            subseq=outer_seq[l:l+W]
            theta_1=init_theta_1(subseq,p)
            
            #compute Pr(Xij/theta1)
            pr_list=[]
            for i,seq in enumerate(seqs):
                    temp=[compute_pr_theta_1(seq, 0,W,theta_1)]
                    for j in range(1,len(seq)-W+1):
                        if l==0:
                            #brute force compute  pr(xij/theta1)
                            temp.append(compute_pr_theta_1(seq, j,W,theta_1))
                        else:
                            #use dp to compute pr(xij/theta1)
                            temp.append(compute_pr_theta_1_dp(seqs,prev_pr_list,theta_1,prev_p,i,j-1,W))
                    pr_list.append(temp)
            prev_p=theta_1 
            prev_pr_list=pr_list
      
            
            #padd the array into same length
            pad= max([ len(i) for i in prev_pr_list])
            
        
            pr_list_padded=np.array([i + [-float('inf')]*(pad-len(i)) for i in prev_pr_list])
            
            #pos is tuple (np.array of sequence id , np.array of pos index)
            pos=prepare_pos(seqs,pr_list_padded,W,model_type=model_type)
                
            for lambi,lambda_dict in enumerate(lambda_list):
                   
                    
                    topn=int(n*lambda_dict[0]*(len(seqs[0])-W+1))
                    if model_type=='OOPS':
                        topn=len(seqs)

                    seq_id_sub=pos[0][:topn]
                    start_list=pos[1][:topn]
                    #print(seq_id_sub,start_list)
                    theta_1=get_theta_1_in_test(seqs,seq_id_sub,start_list,W ,epsilon=epsilon) 
                    
                    likelihood=compute_likelihood(seqs,W,theta_0,theta_1,seq_id_sub,start_list,lambda_dict, model_type=model_type)
                   
                    if max_likelihood[lambi]<likelihood:
                        
                      
                        max_likelihood[lambi]=likelihood
                        #max_subsequence[lambi]=subseq
                        max_theta1_from_test[lambi]=theta_1
                        max_pos=tuple([seq_id_sub,start_list])
                        max_z[lambi]=get_indicator_z(seqs,max_pos)
               
                    
    return max_theta1_from_test,max_z


def get_lambda_list(seqs,W,model_type='OOPS'):
    lambda_list=[]
    #mean
    mm=sum([len(seq)-W+1 for seq in seqs])/len(seqs)
    n=len(seqs)
    if model_type=='OOPS':
        lambda_dict={}
        for i,seq in enumerate(seqs):
            
            m=len(seq)-W+1
            lambda_=1/m
            lambda_dict[i]=lambda_
        lambda_list.append(lambda_dict)

    else:
        lambda_min=[1/(n**0.5)/(len(seq)-W+1) for seq in seqs]
        if model_type=='ZOOPS': 
            lambda_max=[1/(len(seq)-W+1) for seq in seqs]
        elif model_type=='TCM':
            lambda_max=[mm/(len(seq)-W+1)/(W+1) for seq in seqs]
        ele=lambda_min
        t=1
        while(ele[0]<lambda_max[0]):
            lambda_dict={ i:ele[i] for i in range(n)}
            ele= [i*2 for i in ele]
            t+=1
            lambda_list.append(lambda_dict)
        lambda_dict={ i:lambda_max[i] for i in range(n)}
        lambda_list.append(lambda_dict)
    return lambda_list


def compute_f0(seq,i,theta_0,lambda_value,W):
    return get_pr_given_q(seq,theta_0)+log(1-lambda_value[i]*(len(seq)-W+1))

def compute_fj(seq, i, j, theta_0,theta_1,W,lambda_value):
    return get_pr_given_theta(seq, j, theta_0,theta_1,W)+log(lambda_value[i])
    
def Estep(seqs,W,theta_0,theta_1,lambda_value, model_type):
        if model_type=='OOPS':
            pr_list=[]      
            for i,seq in enumerate(seqs):
                    temp=[]

                    for j in range(0,len(seq)-W+1):
                        temp.append(get_pr_given_theta(seq, j, theta_0,theta_1,W))
                    pr_list.append(temp)

            z_expected=[list(np.exp(p_sub-sumlogProbList(p_sub))) for p_sub in pr_list]
        elif model_type=='ZOOPS':
            pr_list=[]    
            for i,seq in enumerate(seqs):
                    temp=[]
                    f0=compute_f0(seq,i,theta_0,lambda_value,W)
                    for j in range(0,len(seq)-W+1):
                        temp.append(compute_fj(seq,i, j, theta_0,theta_1,W,lambda_value))
                    temp=list(np.exp(temp-sumlogProbList(temp+[f0])))
                    pr_list.append(temp)
            z_expected=pr_list
        else:
            pr_list=[]    
            for i,seq in enumerate(seqs):
                    temp=[]
                    for j in range(0,len(seq)-W+1):
                        z1=np.exp(compute_pr_theta_1(seq, j,W,theta_1))*lambda_value[i]
                        z2=np.exp(compute_pr_theta_0(seq, j,W,theta_0))*(1-lambda_value[i])
                        #temp.append(np.exp(z1-sumlogProbList([z1,z2])))
                        temp.append(z1/(z1+z2))
                        
                       # print(np.exp(compute_pr_theta_1(seq, j,W,theta_1)),np.exp(compute_pr_theta_0(seq, j,W,theta_0)), lambda_value[i],np.exp(z1-sumlogProbList([z1,z2])))
                    
                    pr_list.append(temp)
            z_expected=pr_list
          
        return z_expected
    
    

def compute_joint_likelihood(seqs,W,theta_0,theta_1,z_list,lambda_value,model_type='OOPS'):
    #start_list=np.array([np.argmax(i) for i in z_list])
    res=0
    if model_type=='OOPS':
        for i,seq in enumerate(seqs):
            res+=log(lambda_value[i])
            for j in range(len(seqs[i])-W+1):               
                res+=z_list[i][j]*get_pr_given_theta(seq,j, theta_0,theta_1,W)
                
    elif model_type=='ZOOPS':
         for i,seq in enumerate(seqs):
                Qi=sum(z_list[i])
    
                res+=(1-Qi)*get_pr_given_q(seq,theta_0)
               
                res+=(1-Qi)*log(1-lambda_value[i]*len(z_list[i]))
                
                res+=Qi*log(lambda_value[i])
      
                for j in range(len(seqs[i])-W+1):               
                    res+=z_list[i][j]*get_pr_given_theta(seq,j, theta_0,theta_1,W)
    else:
          for i,seq in enumerate(seqs):
            for j in range(len(seqs[i])-W+1): 
                res+=(1-z_list[i][j])*compute_pr_theta_0(seq, j,W,theta_0)
                res+=z_list[i][j]*compute_pr_theta_1(seq, j,W,theta_1)
                res+=(1-z_list[i][j])*log(1-lambda_value[i])
                res+=z_list[i][j]*log(lambda_value[i])

    return res
        
def update_U(U,Z,W):
    for i in range(len(U)):
        for j in range(len(U[i])):
            temp_list= Z[i][max(j-W+1,0):j+1]
            U[i][j]=U[i][j]*(1-max(temp_list))
    return U

def get_pr_V(U,W):
    V=[[0  for _ in range(len(Ui)-W+1)]for Ui in U]
    for i in range(len(U)):
        for j in range(len(U[i])-W+1):
            temp_list= U[i][j:j+W]
            V[i][j]=min(temp_list)
    
    return V    


def update_lambda(z_list):
    lambda_value={}
    n=len(z_list)
    for i in range(len(z_list)):
        lambda_value[i]=0
        m=len(z_list[i])
        for j in range(len(z_list[i])):
            lambda_value[i]+=z_list[i][j]
        lambda_value[i]=lambda_value[i]/m
            
    return lambda_value


def get_theta_different(prev,cur):
    if prev is None:
        return float('inf')

    return np.sum(np.abs(np.exp(prev)-np.exp(cur)))

def init_theta_0(seqs):
    theta_0=np.zeros(len(dict_map.keys()))
    total=sum( [len(seq) for seq in seqs])
    for seq in seqs:
        for ele in seq:
            theta_0[dict_map[ele]]+=1
    return log(theta_0/total),theta_0

def get_null(cnt_dict):
    k=cnt_dict/sum(cnt_dict)
    return sum(k*np.log(k))*sum(cnt_dict)
def update_dict(cnt_dict, tartget_motifs):

    for motif in  tartget_motifs:
        try:
            for site in motif[0]:
                cnt_dict[dict_map[site]]-=1
        except:
            pass
    return cnt_dict




def EM(seqs,W,theta_0_from_test,theta_1_from_test,U,epsilon,lambda_value,model_type):
    
    theta_0=theta_0_from_test
    theta_1=theta_1_from_test
    max_theta1=None
    max_likelihood=-float('inf')
    max_z=None
    max_subsequence=None
    nc=compute_nc(seqs)
    prev=None
    threshold=0.000001
    t=0
    
    V=get_pr_V(U,W)

    
    for kk in range(200):
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

   
        if  get_theta_different(prev,theta_1)<threshold:
            break
#         if prev is not None and likelihood>prev and likelihood-prev<0.0001:
#             print(kk,likelihood-prev)
#             break
      
        prev=theta_1
    
    return  likelihood,z_expected,theta_1,theta_0,t


def get_target_z(seqs,z_list,W,model_type='OOPS'):
    n=len(seqs)
    pos=prepare_pos(seqs,z_list,W,model_type=model_type)
    
    if model_type=='OOPS':
        topn=n
        seq_id_sub=pos[0][:topn]
        start_list=pos[1][:topn]
        max_pos=tuple([seq_id_sub,start_list])
    elif model_type=='ZOOPS':
        seq_id_sub=[]
        start_list=[]
        id_sub=pos[0]
        s_list=pos[1]
        for i,idx in enumerate(id_sub):
            if z_list[idx][s_list[i]]>=0.5:
                seq_id_sub.append(idx)
                start_list.append(s_list[i])
            else:
                break
        max_pos=tuple([seq_id_sub,start_list])                
   
    max_z=get_indicator_z(seqs,max_pos)
    return max_z
    
    
def majority(motifs):
    motif_flatten=[]
    for motif in motifs:
        motif_flatten+=motif
   
    motif_sequences=np.array([list(s) for s in motif_flatten])

    from scipy import stats
    try:
        res=stats.mode(np.array(motif_sequences))[0][0]
        return "".join(res)
    except:
        
        return "" 

def get_motif_seqs(seqs,z_list,W):
    motifs=[]
    for i,z in enumerate(z_list):
        if sum(z)==0:
            temp=[]
            motifs.append(temp)
        else:
            temp=[]
            index_list=np.where(z)[0]
            for k in index_list:

                temp.append(seqs[i][k:k+W])
            motifs.append(temp)
    return motifs
  
def majority(motifs):
    motif_flatten=[]
    k=len(motifs[0])
    for motif in motifs:
        if k!=len(motif):
            continue
        motif_flatten+=motif
   
    motif_sequences=[list(s) for s in motif_flatten]
   
    from scipy import stats
    try:
        res=stats.mode(np.array(motif_sequences))[0][0]
        return "".join(res)
    except:
        return ""

def get_motif_p_value(sequence, theta_0,theta_1,W):
    

    res = 0
    for i in range(len(sequence)):  
            res += theta_1[ dict_map[sequence[i]],i]
            res -= theta_0[dict_map[sequence[i]]]

    LR=2*(res)
    p = chi2.sf(LR, 1)# L2 has 1 DoF more than L1
    return p
def  recover_from_pwm(x):
    result=np.argmax(x,axis=0)
    output=''
    for i in result:
        output+=inverse_dict_map[i]
    return output
 
    
def get_char_stat(pred,truth,seq_len):
    if truth!=[]:
         gold_motif_list=  [t.sites for t in truth]
    else:
         gold_motif_list=  [" "]
    
    pred_motf=pred.sites
   
    s1=pred_motf
    best_target=None
    best_match=-float('inf')
    for s2 in gold_motif_list:
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
        max_match=len(s1[pos_a:pos_a+size])
      
        if best_match< max_match:
            best_match=max_match
            best_target=s2
    tp=best_match
    
    fp=len(s1)-best_match
    fn=len(best_target)-best_match
    tn=seq_len-(len(s1)+len(best_target)-best_match)

    precsion=tp/(len(s1))
    recall=tp/(len(best_target))
        
    return precsion,recall
def get_roc(tpp,fpp):
    tpp=[0]+ tpp+[1]
    fpp=[0]+ fpp+[1]    
    fpp_=np.array(fpp)
    fppx=np.array([0]+fpp[:-1])
    fpp=list(fpp_-fppx)
    roc=np.trapz( tpp,fpp)
    if roc<0:
        return 0
    return roc

def get_word_stat(mp,ipass,truth,W,shift=2):  
    pred=mp.start[ipass]
    seqs=mp.seqs
    tp=0
    fp=0
    fn=0
    break_flag=0
    tn=0
    for i,t_motifs in enumerate(truth):
        if pred[i]==[] and len(t_motifs)==0:
            tp+=1
            tn+=len(seqs[i])
            continue
        if len(t_motifs)==0:
            if pred[i]!=[]:
                fp+=1
                continue
        if pred[i]==[]:
            fn+=1
            continue
        
        for t_motif in t_motifs:

            t=t_motif.pos[1]
            t_w=t_motif.length
           
            if pred[i][0]>= t-shift and pred[i][0]+W<= t+t_w+shift:
                tp+=1
                tn+=(len(seqs[t_motif.pos[0]])-t_motif.length+1)-1
                break_flag=1
                continue
        
        if break_flag==1:
            break_flag=0
            continue
        
        fp+=1
        
    acc=tp/len(seqs)

    tpp=acc
    
    fpp=fp/(fp+tn)
    if fp==0:
        fpp=0
    
    return acc,tpp,fpp
    
def update_statistics(mp,mt):
    npass=len(mp.motifs)

    
    for ipass in range(npass):

        pp=[]
        rr=[]
        for motif in mp.motifs[ipass]:
            if motif==[]:
                continue
                
            update_dict={}
            (seqid,startid)=motif.pos


            update_dict['loglikelihood']=get_pr_given_theta(mp.seqs[seqid], startid, motif.theta_0,motif.theta_1,motif.length)
            update_dict['p_value']=get_motif_p_value(motif.sites, motif.theta_0,motif.theta_1,motif.length)

            char_precsion,char_recall=get_char_stat(motif,mt.motifs[seqid],len(mp.seqs[seqid]))

            pp.append(char_precsion)
            rr.append(char_recall)
           
            update_dict['precsion']=char_precsion
            update_dict['recall']=char_recall
            motif.update_statistics(**update_dict)

        acc,tpp,fpp=get_word_stat(mp,ipass,mt.motifs,len(mp.consensus_motif[ipass]),shift=4)
        mp.motif_acc[ipass]= acc

       
        mp.motif_roc[ipass]=get_roc([tpp],[fpp])
        
        mp.char_precision[ipass]=np.mean(pp)
        mp.char_recall[ipass]=np.mean(rr)
    return 

def get_consensus_from_p(x):
    return "".join(np.array(elements)[np.argmax(x,axis=0)])