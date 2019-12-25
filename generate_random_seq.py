import random
elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}


def generate_random_sequence(mt,n_random_seqs):
    mean=np.mean(list(map(len,mt.sequences)))
    var=np.var(list(map(len,mt.sequences)))
    
    for k in range(n_random_seqs):
        print(">random_{}".format(k+1))
        t=np.random.sample(1)*var
        if t<0:
            length=int(mean-np.sqrt(t))
        
        else:
            length=int(mean+np.sqrt(t))

        s=''
        for i in range(length):
            s+= inverse_dict_map[np.random.randint(4)]

    return 
        
