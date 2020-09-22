# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import scipy as sp
import scipy.optimize
import math

#Find optimal solution given adj matrix and covariate matrices

def solve_opt_policy(adj, cov_mat, coef_vec, deploy_max):  
    n=((adj).shape)[1]
    inv_component=np.linalg.inv(np.diagflat(adj.dot(np.ones(n))))
    first_component=np.dot(coef_vec.dot(cov_mat),inv_component)
    multval=np.dot(first_component,adj)
    A_mat=np.identity(n)
    A_mat=np.vstack([A_mat,np.ones(n)])
    b_vec=np.ones(n)
    b_vec=np.append(b_vec,deploy_max)
    soln=sp.optimize.linprog(c=-multval, A_ub=A_mat, b_ub=b_vec)
    opt_policy=np.rint(soln['x'])
    opt_val=soln['fun']
    return(dict(value=opt_val,policy=opt_policy))


## Knapsack Greedy Heuristic
## Use determinant idea, instead of largest eigenvalue

def knapsack_greedy(adj, cov_mat, cost_max):
    n=((adj).shape)[1]
    exposed_vec=np.zeros(n)
    treated_choice_vec=np.zeros(n)
    var_reduction=np.zeros(n)
    values=np.zeros(n)
    choice_seq=list()
    cand_experiment=111111111
    cand_val=-math.inf
    for i in range(n):
        covs_node=cov_mat[:,np.array(adj[i,],dtype=bool)[0]]
        var_reduction[i]=1/np.trace(np.linalg.inv(np.dot(covs_node,covs_node.T)))
        values[i]=var_reduction[i]/np.sum(adj[i,])
        if(np.sum(adj[i,])<=cost_max):
            if(values[i]>cand_val):
                cand_val=values[i]
                cand_experiment=i
                cand_covs_node=cov_mat[:,np.array(adj[i,],dtype=bool)[0]]
    treated_choice_vec[cand_experiment]=1
    choice_seq.append(cand_experiment)
    exposed_vec=np.maximum(exposed_vec,adj[cand_experiment,])
    covs_node=cand_covs_node
    k=1
    while (np.sum(exposed_vec)<=cost_max):
        cand_experiment=111111111
        cand_val=-math.inf
        var_reduction=np.zeros(n)
        values=np.zeros(n)
        old_var=np.trace(np.linalg.inv(np.dot(covs_node,covs_node.T)))
        for i in range(n):
            new_exp=np.maximum(exposed_vec,adj[i,])
            if(treated_choice_vec[i]==0 and np.sum(new_exp)<=cost_max):
                cand_covs_node=cov_mat[:,np.array(new_exp,dtype=bool)[0]]
                new_var=np.trace(np.linalg.inv(np.dot(cand_covs_node,cand_covs_node.T)))
                values[i]=(old_var-new_var)/(np.sum(new_exp)-np.sum(exposed_vec))
                if(values[i]>cand_val):
                    cand_val=values[i]
                    cand_experiment=i
                    cand_covs_node=cov_mat[:,np.array(adj[i,],dtype=bool)[0]]
        covs_node=cand_covs_node
        if(cand_experiment!=111111111):
            choice_seq.append(cand_experiment)
            treated_choice_vec[cand_experiment]=1
            exposed_vec=np.maximum(exposed_vec,adj[cand_experiment,])
            k=k+1
        else:
            break
    return(np.array(choice_seq))
    


## Learning the coefficient vector through the given sequence

def learn_gamma(adj,cov_mat,lambda_vec,seq):
    n=((adj).shape)[1]
    k=np.shape(cov_mat)[0]
    inv_component=np.linalg.inv(np.diagflat(adj.dot(np.ones(n))))
    total_outcomes=np.zeros(30*n)
    total_covs=np.zeros((k,30*n))
    total_exps=np.zeros(30*n)
    l=0
    for node in seq:
        treated=np.zeros(n)
        treated[node]=1
        expose=np.dot(treated,adj)
        exp_exposed=np.dot(inv_component,expose.T)
        exp_exposed=exp_exposed[np.array(adj[node,],dtype=bool)[0]]
        covs_exposed=cov_mat[:,np.array(adj[node,],dtype=bool)[0]]
        num_exposed=np.shape(covs_exposed)[1]
        outcome_exposed= np.dot(np.dot(lambda_vec,covs_exposed),exp_exposed)+np.random.normal(size=num_exposed)
        total_covs[:,l:(l+num_exposed)]=covs_exposed*exp_exposed
        total_exps[l:(l+num_exposed)]=(exp_exposed).ravel()
        total_outcomes[l:(l+num_exposed)]=(outcome_exposed).ravel()
        l=l+num_exposed
    total_outcomes=total_outcomes[0:l]
    total_exps=total_exps[0:l]
    total_covs=total_covs[:,0:l]
    gamma_hat= np.dot(np.dot(np.linalg.inv(np.dot(total_covs,total_covs.T)),total_covs),total_outcomes)
    return(dict(total_out=total_outcomes,total_exposure=total_exps,total_covariates=total_covs,gamma=gamma_hat))
    
   


def simul_numbers(num_sim,graph_size,graph_connect,cov_num,max_deploy,max_learn):
    value_gap=np.zeros(num_sim)
    for i in range(num_sim):
        rand_graph = nx.erdos_renyi_graph(graph_size,graph_connect)
        mat_rand_graph=nx.adjacency_matrix(rand_graph,weight=None)
        mat_rand_graph.setdiag(1)
        mat_adj_graph=(mat_rand_graph.todense())  
        covariate_mat = np.random.multivariate_normal(np.zeros(cov_num), np.identity(cov_num)*2, graph_size).T
        true_vec = np.random.multivariate_normal(np.zeros(cov_num), np.identity(cov_num)*3)
        try:
            treated_seq=knapsack_greedy(mat_adj_graph,covariate_mat,max_learn)       
            learn_results=learn_gamma(mat_adj_graph,covariate_mat,true_vec,treated_seq) 
            learned_coef=learn_results['gamma']
            learned_prob=solve_opt_policy(mat_adj_graph,covariate_mat,learned_coef,max_deploy)
            val_learned=learned_prob['value']
            oracle_prob=solve_opt_policy(mat_adj_graph,covariate_mat,true_vec,max_deploy)
            val_oracle=oracle_prob['value']
            value_gap[i]=1-abs(val_learned/val_oracle)
        except:
            value_gap[i]=value_gap[i-1]
            print('Exception')
        print(i)
    return(value_gap)
    
    
def simul_grid(graph_size,graph_connect,num_sims):
    max_deploy_grid=np.linspace(start=graph_size, stop=graph_size, num=1)
    cov_num=5
    mean_gap_grid=np.zeros(shape=(1,4))
    sdev_gap_grid=np.zeros(shape=(1,4))
    i=0
    for j in max_deploy_grid:
        k=0
        max_learn_grid=np.linspace(start=j/4, stop=3*j/4, num =4)
        for q in max_learn_grid:
            result_gaps=simul_numbers(num_sims,graph_size,graph_connect,cov_num,max_deploy=j,max_learn=q)
            mean_gap_grid[i,k]=np.mean(result_gaps)
            sdev_gap_grid[i,k]=np.std(result_gaps)
            k=k+1
        i=i+1
    return(dict(mean_gaps=mean_gap_grid,sdev_gaps=sdev_gap_grid))
                
                        
                
result_sims=simul_grid(500,0.1,500)


np.savetxt('sim_results_n1000_mean.csv', result_sims['mean_gaps'], delimiter=',')
np.savetxt('sim_results_n1000_sdev.csv', result_sims['sdev_gaps'], delimiter=',')

