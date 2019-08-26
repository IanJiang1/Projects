import numpy as np
import multinomial_rnd
from mLogProb import *
from valid_RandIndex import valid_RandIndex
from multinomial_rnd import *
from dirichlet_rnd import *

def mGibbs(adj, K, alpha=None, numIter=None, zTrue=None, zInit=None):
    """
    Gibbs sampler
    :param adj: NxN adjacency matrix for observed graph,
                where negative entries indicate missing observations
    :param K: number of communities
    :param alpha: Dirichlet concentration parameter
    :param numIter: number of Gibbs sampling iterations
    :param zInit: 1xN vector of initial community assignments
    :return: z: 1xN vector of final community assignments
            pi: Kx1 vector of community membership probabilities
            W: KxK matrix of community interaction probabilities
    """
    N = adj.shape[0]
    mask = (np.ones((N, N)) - np.identity(N)) > 0
    mask = np.multiply(mask, (adj >= 0))
    if alpha is None:
        alpha = 1.0
    if numIter is None:
        numIter = 1
    if zTrue is None:
        zTrue = np.ones(N, dtype=int)
    if zInit is None:
        z = multinomial_rnd(np.ones(K), N) 
        z = z.astype(int) - 1 
    else:
        z = zInit
    #z = np.array([0,0,2,1,2,2,2,2,2,0,1,2,1,0,0,1,2,1,1,2,0,2,1,1,1,2,0,0,2,2])

    pi = np.random.uniform(0,1, K)
    pi = pi/np.sum(pi)
    W = np.random.uniform(0, 1, (K,K))
    logProb = np.zeros(numIter)
    randI = np.zeros(numIter)
    print('SB Gibbs: ')
    for tt in range(numIter):
        # TODO: sample mixture probabilities pi, pi = ?
        #================================================
        #Computing updated Dirichlet posterior parameters
        alpha_upd = alpha + np.array([(z == i).sum() for i in range(K)]).astype(int)
        #print("alpha_upd = {}".format(alpha_upd))
        pi = dirichlet_rnd(alpha_upd, 1).flatten()             

        # TODO: sample community interaction parameters W
        #================================================
        for kk in range(K):
            for ll in range(K):

                #IJIANG6: restricting attention only to nodes in relevant communities
                conn = adj[z == kk,:][:, z == ll]
                a = conn.sum() + 1
                
                #Removing self-edges from consideration
                if kk != ll:
                    b = conn.size - conn.sum() + 1
                else:
                    b = conn.size - conn.shape[0] - conn.sum() + 1
                #print("a, b: {}".format((a, b)))

                #IJIANG6: drawing from posterior conditional on connection probabilites
                W[kk,ll] = dirichlet_rnd([a, b], 1)[0]

        #print("Connection Probabilities: {}".format(W))
                
        # TODO: sample community assignments z in random order
        #================================================
        
        #for ii in range(N):	
        for ii in np.random.permutation(N):	
            #IJIANG6: Obtaining product over all relevant connection probabilities
            row = adj[ii, :]
            col = adj[:, ii]
            #p = np.array([W[kk, z[row > 0] - 1].prod() * W[z[col > 0] - 1, kk].prod() for kk in range(K)], dtype = float)
            p = np.array([pi[kk]*(1 - W[z[col == 0], kk]).prod() * (1 - W[kk, z[row == 0]]).prod() * W[kk, z[row > 0]].prod() * W[z[col > 0], kk].prod() / ((1 - W[kk, kk])**2) for kk in range(K)], dtype = float)
            #print(W[kk, z[row > 0]])
            z[ii] = multinomial_rnd(p, 1) - 1	

        logProb[tt] = mLogProb(adj, z, pi, W, alpha)
        AR, RI, MI, HI = valid_RandIndex(z, zTrue)
        randI[tt] = RI
        #print("pi_alpha: " + str(pi))
        #print("a, b = " ,a, b)
        if tt % round(numIter / 10) == 0:
            print('.')
    return [z, pi, W, logProb, randI]
