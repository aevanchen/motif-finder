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
import pandas as pd

elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}
def find_most_matched_motif(pred,truth):

    if truth!=[]:
        gold_motif_list=  [t.sites for t in truth]
        gold_start= [t.pos[1] for t in truth]
    else:
         gold_motif_list=  [" "]
    
    pred_motf=pred.pos[1]
   
    score=[]
    for s2 in gold_start:
        score.append(abs(s2-pred_motf))
    k=np.argmin(score)
    
    return gold_motif_list[k]
def print_meme_result(mp,mtruth):
    print("MEME {} Model Result for DNA Sequence {}".format(mp.model_type,mp.motif_file))
    for i in range(mp.npass):
        print("PASS {}".format(i+1))
        motifs=mp.motifs[i]
        print ('{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}'.format("Sequence ID","Sequence Name","Pred Sites","Pred Width",'Matched True Sites',\
                                                                                     'True Width',"Loglihood","P_value","Precsion",'Recall','Z value'))
        empty_list=[]
        for j,motif  in enumerate(motifs):
            if motif==[]:
                empty_list.append(j)
                continue

            if motif.precsion>=0.1:
                truth_site=find_most_matched_motif(motif,mtruth.motifs[j])
                truth_w=len(truth_site)

            else:
                truth_w='N/A'
                truth_site='N/A'    

            print ('{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}'.format(motif.pos[0]+1,motif.seq_name,motif.sites,motif.length, truth_site,\
                                                               truth_w, round(motif.loglikelihood,3),round(motif.p_value,10),motif.precsion,motif.recall,motif.z_value))
        for idx in empty_list:
            print("Sequence {} - {} DOES NOT Contain Motif".format(idx+1,mp.seqname[idx]))
        print("Sequence Level Evaluation Metric")
    #     print ('{:<25}{:<25}{:<25}{:<25}{:<25}'.format("ROC","Motif Match Precsion","Motif Match Recall","Site Match Precsion",'Site Match Recall'))

        print("Consensus motif:  {}".format(get_consensus_from_p(mp.theta_1_list[i])))
        print ('{:<25}{:<25}{:<25}{:<25}'.format("roc","acc","char precision","char recall"))
        print( '{:<25}{:<25}{:<25}{:<25}'.format(mp.motif_roc[i],mp.motif_acc[i],mp.char_precision[i],mp.char_recall[i]))
        print('*****************************************************************************************************************************************') 
        
def describute_data(filename):
    mtruth=MotifTruth(filename)
    col=["Sequence ID","Sequence Name","True Width","Start","True Sites I","True Sites 2",'True Sites 3']

    df = pd.DataFrame(columns=col)
    for i,motifs in enumerate(mtruth.motifs):
        if motifs==[]:
            tw=""
        else:
            tw=motifs[0].length
        try:
            start=", ".join(map(str,mtruth.start[i]))
        except:
            start=""
        true_site=[]
        for j in range(3):
            try:
                true_site.append(motifs[j].sites)
            except:
                true_site.append(" ")

        df = df.append( {col[kk]:value for kk, value in enumerate([i+1,mtruth.seqname[i],tw,start,true_site[0],true_site[1],true_site[2]])}, ignore_index=True)   

    print("True consensus motif : {}".format(mtruth.consensus_motif))
    latex=df.to_latex(index=False)
    return latex


def generate_latex(mp,mt,print_latex=False):
#     print(" \\begin{table}\n\
# \\setlength\tabcolsep{4pt} % default value is 6pt\n\
# \\footnotesize\n\
#  \\caption{MEME OOPS model results for the synthetic toy dataset }\n\
#  \\label{sample2}\n\
# \\centering")
    
    for i in range(mp.npass):

        motifs=mp.motifs[i]
        col=[" ID"," Name","Pred Sites","Width",'True Sites',\
                                                                                         'True Width',"Loglihood","P_value","Precsion",'Recall']
        df = pd.DataFrame(columns=col)
        empty_list=[]
        for j,motif  in enumerate(motifs):
            if motif==[]:
                empty_list.append(j)
                df = df.append( {col[i]:value for i, value in enumerate([j+1,mp.seqname[j],"N/A", "N/A", "N/A",\
                                                               "N/A", "N/A","N/A","N/A","N/A"])}, ignore_index=True)   
                continue

            if motif.precsion>=0.1:
                truth_site=find_most_matched_motif(motif,mt.motifs[j])
                truth_w=len(truth_site)
              
            else:
                truth_w='N/A'
                truth_site='N/A'    

            df = df.append( {col[i]:value for i, value in enumerate([motif.pos[0]+1,motif.seq_name,motif.sites,motif.length, truth_site,\
                                                               truth_w, round(motif.loglikelihood,3),round(motif.p_value,7),round(motif.precsion,3),round(motif.recall,3)])}, ignore_index=True)   
        if not print_latex:
            print(df)
        else:
            latex=df.to_latex(index=False)
        
            print(latex)
        
       # break
        print()
        for idx in empty_list:
            print("Sequence {} - {} DOES NOT Contain Motif".format(idx+1,mp.seqname[idx]))
            print("Sequence Level Evaluation Metric")
    #     print ('{:<25}{:<25}{:<25}{:<25}{:<25}'.format("ROC","Motif Match Precsion","Motif Match Recall","Site Match Precsion",'Site Match Recall'))

        print("Consensus motif:  {}".format(get_consensus_from_p(mp.theta_1_list[i])))
        print ('{:<25}{:<25}{:<25}{:<25}'.format("roc","acc","char precision","char recall"))
        print( '{:<25}{:<25}{:<25}{:<25}'.format(mp.motif_roc[i],mp.motif_acc[i],mp.char_precision[i],mp.char_recall[i]))
        print('*****************************************************************************************************************************************') 
        print('******************************************************************************************************************************')
#     print("\\end{table}")
        