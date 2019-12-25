
from utlis import *

from collections import defaultdict


elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}



class MotifTruth():
    def __init__(self, filename):
        self.motif_file=filename
        self.sequences=[]
        self.start=[]
        self.seqname=[]
        self.motif_list=[] #flattened last of motif
        self.motifs=[]
        self.read_motif()
        self.consensus_motif=majority( [[m.sites] for m in  self.motif_list])
        
        
    def read_motif(self):
        sequences=[]
        seqname=[]
        motif_list=[]
        motifs=[]
        true_start=[]

        with open(self.motif_file, "r") as f:
            s=""
            seq_id=0    
            start=[]
            width=[]
            temp_motif=[]
            temp_start=[]
            for l in f.readlines():
                    if l.strip()[0] == ">":

                        if s!="":
                            for i,ss in enumerate(start):
                
                                new_motf=Motif(width[i],s[ss-1:ss-1+width[i]],(seq_id,ss-1), name)
                                temp_start.append(ss-1)
                                motif_list.append(new_motf)
                                temp_motif.append(new_motf)

                            motifs.append(temp_motif)
                            true_start.append(temp_start)
                            temp_motif=[]
                            temp_start=[]

                        info=l.strip().split(" ")
                        name=info[0][1:]
                        seqname.append(name)
                        start=[]
                        width=[]

                        for j,content in  enumerate(info[1:]):
                            if j%2==0:
                                start.append(int(content))
                            else:
                                width.append(int(content))

                        if s=="":continue
                        sequences.append(s)
                        seq_id+=1               
                        s=""

                    else:
                        s += l.strip().upper()

            for i,ss in enumerate(start):
                new_motf=Motif(width[i],s[ss-1:ss-1+width[i]],(seq_id,ss), name)
                motif_list.append(new_motf)
                temp_start.append(ss-1)
                temp_motif.append(new_motf)

            motifs.append(temp_motif)
            true_start.append(temp_start)             
            sequences.append(s)
           
            assert(len(motifs)==len(sequences)==len(seqname)==len(true_start))
            self.sequences=sequences
            self.start=true_start
            self.seqname=seqname
            self.motif_list=motif_list #flattened last of motif
            self.motifs=motifs
            return 
    def get_motif_sites(self):
        output=[]
        temp=[]
        for motif_seq in self.motifs:
            for motif in motif_seq:
                temp.append(motif.sites)
            output.append(temp)
            temp=[]
        return output
    
class Motif():
    def __init__(self, length,sites, pos,seq_name,theta_1=None,theta_0=None):
        self.length=length
        self.sites=sites
        self.pos=pos
        self.seq_name=seq_name
        self.theta_1=theta_1
        self.theta_0=theta_0
        self.precsion=None
        self.recall=None
        self.loglikelihood=None
        self.p_value=None
        self.z_value=None
    def update_statistics(self,**kwargs):
        for k in kwargs.keys():
              self.__setattr__(k, kwargs[k])
        return 
    
    
class MotifPreditor(): 
    def __init__(self, filename,**kwargs):
        self.motif_file=filename
        self.seqs=[]
        self.seqname=[]
        self.read_fasta()
        for k in kwargs.keys():
              self.__setattr__(k, kwargs[k])
        
        self.start=[[[] for i in range(len( self.seqs))] for j in range(self.npass)]
    
        self.motifs=[ [[] for i in range(len( self.seqs))] for j in range(self.npass)]
        self.consensus_motif=[[] for j in range(self.npass)]
        
        self.test_motif=[[] for j in range(self.npass)]

        self.converged_motif=[[] for j in range(self.npass)]
        self.char_precision=[[] for j in range(self.npass)]
        self.char_recall=[[] for j in range(self.npass)]
        self.motif_roc=[[] for j in range(self.npass)]
        self.motif_acc=[[] for j in range(self.npass)]
       
        self.theta_1_list=[[] for j in range(self.npass)]
     
        
    def read_fasta(self):
        with open(self.motif_file, "r") as f:
            sequences = []
            seqname=[]
            s = ""
            for l in f.readlines():
                if l.strip()[0] == ">":
                    info=l.strip().split(" ")
                    name=info[0][1:]
                    seqname.append(name)
                    # skip the line that begins with ">"
                    if s == "": continue
                    sequences.append(s)
                    s = ""
                # keep appending to the current sequence when the line doesn't begin
                # with ">"
                else:
                    s += l.strip().upper()
            sequences.append(s)
            self.seqs=sequences
            self.seqname=seqname
     
            return
    
    def MEME(self):

        p=get_p(self.initial_beta)  #the Dirichlet prior

        npass=self.npass

        Wmin=self.Wmin

        max_motif_length= int(0.8* min([len(seq) for seq in self.seqs]))

        Wmax=min(self.Wmax,max_motif_length) 

        U= [[1  for i in range(len(seq))]for seq in self.seqs]

        theta_0_from_test,cnt_dict=init_theta_0(self.seqs)
        #theta_0_from_test=log(np.array([0.25,0.25,0.25,0.25]))

        test_dict={}

        Wlist=range(Wmin,Wmax+1)
       # Wlist=[5,6,8]
        for W in Wlist:
                lambda_list=get_lambda_list(self.seqs,W,model_type=self.model_type)
                max_theta1_from_test,max_z=test(self.seqs,W,lambda_list,p,theta_0_from_test,self.beta,quick_test=self.quick_test,model_type=self.model_type)
                test_dict[W]=[max_theta1_from_test,lambda_list,max_z]
                
        
        

        for ipass in range(npass):
            min_G=float('inf')
            target_consensus_motif=None
            target_W=None
            target_z=None
            target_theta_test=None
            target_theta_1=None
            target_theta_0=None
            target_z_expected=None
            tartget_motifs=None
            tract_test_dict=defaultdict(list)

            for W in tqdm(Wlist):
               
         
                    lambda_list=test_dict[W][1]
                    
             
                    
                    for lambda_i,lambda_value in enumerate(lambda_list):

                        theta_1_from_test=test_dict[W][0][lambda_i] 

                        
                        likelihood,z_list,theta_1,theta_0,total_iters=EM(self.seqs,W,theta_0_from_test,theta_1_from_test,U,self.beta,lambda_value,self.model_type)  
                        tract_test_dict[W].append([lambda_value,get_consensus_from_p(theta_1_from_test),get_consensus_from_p(theta_1),total_iters]) 
                        
                        null_likelihood=get_null(cnt_dict)
                        
                        G=compute_G(likelihood,null_likelihood,W,ipass,model_type=self.model_type)
                        if G<min_G:
                            min_G=G
                          

                            target_W=W
                            target_z_expected=z_list
                         
                            pad= max([ len(i) for i in z_list])

                            pr_list_padded=np.array([i + [-float('inf')]*(pad-len(i)) for i in z_list])
                            
                            
                            
                            target_z=get_target_z(self.seqs,pr_list_padded,W,model_type=self.model_type)
                         
                            
                            #tartget_motifs=get_motif_seqs(seqs,target_z,W)

                            #target_consensus_motif=majority(tartget_motifs)

                            target_theta_1=theta_1
                            target_theta_0=theta_0
                            target_theta_test=theta_1_from_test
            self.theta_1_list[ipass]=target_theta_1
            self.converged_motif[ipass]=tract_test_dict

            #update U
            def set_parameters(z_list,z_expected,ipass,W,theta_1,theta_0,theta_test):         
                    ip=ipass
                    #print(self.test_motif,ip)
                    self.test_motif[ip]=theta_test
                    temp_motif=[]
                    temp_start=[]
                   
                    for i,z in enumerate(z_list):
                        if sum(z)==0:
                            self.start[ip][i]=[]
                          
                            continue
                        else:

                            k=np.where(z)[0][0]
#                             print(i,k,W,self.seqname[i])
#                             print(self.seqs[i])
#                             print(i,k,W,self.seqs[i][k:k+ W])
                            
                            self.motifs[ip][i]=Motif(W,self.seqs[i][k:k+ W],(i,k),self.seqname[i],theta_1=theta_1,theta_0=theta_0)
                            self.motifs[ip][i].update_statistics(**{"z_value":z_expected[i][k]})

                            self.start[ip][i]=[k]
                    
                    self.consensus_motif[ip]=majority([[self.motifs[ip][i].sites] for i in range(len(self.seqs))  if self.motifs[ip][i]!=[]])
                    return
            
            set_parameters(target_z,target_z_expected,ipass,target_W,target_theta_1,target_theta_0,target_theta_test)       
          
            #cnt_dict=update_dict(cnt_dict, [[self.motifs[ipass][i].sites] for i in range(len(self.seqs))  if self.motifs[ipass][i]!=[]])
         
            U=update_U(U,target_z_expected,target_W) 
            theta_0_from_test=cnt_dict/np.sum(cnt_dict)

        return 
   