import numpy as np
import math
# Receptive field creation
class SUSTAIN:
    def __init__(
        self,
        r: float = 2.844642,
        beta: float = 2.386305,
        d: float = 12.0,
        eta: float = 0.09361126,
        tau: float = 0.5,
        supervised: bool = True,
        queried_dim: int = -1,
    ):
        self.r = r
        self.beta = beta
        self.d = d
        self.eta = eta
        self.tau = tau
        self.supervised = supervised
        self.queried_dim = queried_dim  # will be resolved after first stimulus

        # These are set after the first call to reset() or the first trial
        self.n_dims: int = 0
        self.dim_sizes: list[int] = []    # number of values per dimension
        self.dim_offsets: list[int] = []  # start index in flat input vector

        # Cluster state (lists grow as clusters are recruited)
        self.n_clusters: int = 0
        self.H_pos: list[np.ndarray] = []   # cluster positions (flat vectors)
        self.lambdas: np.ndarray = np.array([])  # tunings per dimension (shape: n_dims)
        self.weights: list[np.ndarray] = []  # weight matrices per cluster
    
    def receptive_field(self, l, mu):
        #Eq. 1
        return np.exp(-l * mu)
    
    def distance(self, I_pos:np.ndarray, H_pos_j:np.ndarray, dim: int)->float:
        #eq. 4
        v_i = self.dim_sizes[dim]
        sl = self.dim_slice(dim)
        diff = I_pos[sl] - H_pos_j[sl]
        return 0.5 *np.sum(np.abs(diff))/v_i

    def cluster_act(self, I_pos:np.ndarray, H_po_j: np.ndarray, known_dims: list[int]) -> float:
        #eq 5
        num=0.0
        den=0.0
        
        for i in known_dims:
            l_i = self.lambdas[i]
            mu_ij = self.distance(I_pos, H_pos_j, i)
            rf=self.receptive_field(l_i, mu_i)
            w=l_i**self.r
            num+= w*rf
            den+=w
        if den==0.0:
            return 0.0
        return num/den
    
    def cluster_output(self, activations: np.ndarray)->np.ndarray:
        #eq 6
        if len(activations)==0:
            return np.array([])
        winner = int(np.argmax(activations))
        h_act_winner = activations[winner]
        
        denom = np.sum(activations)
        if denom>0:
            H_out[winner]=H_act_winner/denom
        return H_out
    
    def output_units(self, H_out:np.ndarray, dim: int)->np.ndarray:
        #eq 7
        v_z = self.dim_sizes[dim]
        C_out=np.zeros(v_z)
        for j in range(self.n_clusters):
            if H_out[j] != 0.0:
                C_out += self.weights[j][dim]*H_out[j]
        return C_out
    
    def response_prob(self, C_out: np.ndarray) -> np.ndarray:
        #eq 8
        scaled = self.d * C_out
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)
        return exp_vals/exp_vals.sum()
    
    def humble_teach(self, C_out:np.ndarray, I_pos: np.ndarray, dim:int)->np.ndarray:
        #eq 9
        sl = self.dim_slice(dim)
        I_dim = I_pos[sl]
        t=np.where(I_dim ==1,
                   np.max(C_out, 1.0)
                   np.min(C_out, 0.0))
        return t
    
    def recruit_cluster(self, I_pos:np.ndarray):
        new_pos = I_pos.copy()
        self.H_pos.append(new_pos)
        
        new_weights = [np.zeros(s,dtype=float) for s in self.dim_sizes]
        self.weights.append(new_weights)
        self.n_clusters += 1
        
    def update_pos(self, winner:int, I_pos:np.ndarray):
        #eq 12
        self.H_pos[winner] += self.eta *(I_pos - self.H_pos[winner])
        
    def update_tuning(self, winner:int, I_pos: np.ndarray, known_dims:list[int]):
        #eq 13
        for i in known_dims:
            mu_ij = self.distance(I_pos, self.H_pos[winner], i)
            l_i = self. lambdas[i] 
            self.lambdas[i] += self.eta * np.exp(-l_i*mu_ij)*(1-l_i*mu_ij)
            self.lambdas[i]=max(self.lambdas[i], 1.0)
            
    def update_weights(self, winner: int, H_out_winner: float, C_out: np.ndarray, 
                       t: np.ndarray, dim: int):
        #eq 14
        mu_w=(t-C_out)*H_out_winner
        self.weights[winner][dim] += selgf.eta *mu_w
    
## Setup

    def resolve_dim(self, n_dims:int)-> int:
        q = self.query_dim
        if q <0:
            q = n_dims+q
        return q
    
    def setup(self, dim_sizes: list[int]):
        self.n_dims = len(dim_sizes)
        self.dim_sizes = list(dim_sizes)
        self.dim_offsets=[]
        offeset=0
        for s in dim_sizes:
            self.dim_offsets.append(offset)
            offset+=s
        self.flat_size = offset
        
        self.lambdas = np.ones(self.n_dims, dtype=float)
        
        self.n_clusters =0
        self.H_pos = []
        self.weights = []
        
        self.query_dim_idx = self.resolve_dim(self.n_dims)
    
    def reset(self, dim_sizes: list[int]):
        self.setup(dim_sizes)
    
## Encoding

    def encode(self, stimulus:list[int]) -> np.ndarray:
        stim_vec = np.zeros(self.flat_size, dtype=float)
        for i, val in enumerate(stimulus):
            if val>=0:
                idx = self.dim_offsets[i]+val
                stim_vec[idx]=1.0
        return stim_vec
    
    def dim_slice(self, dim:int):
        start = self.dim_offsets[dim]
        end = start + self.dim_sizes[dim]
        return slice(start, end)
                
## Interface

    def present_stimulus(self, stimulus: list[int], 
                         query_dim: Optional[int] = None)-> dict:
        if self.n_dims ==0:
            dim_sizes = []
            for v in stimulus:
                if v < 0:  #makes sure system is reset
                    raise ValueError('Call reset before presenting stimuli, or\n ensure the first stimulus has known dimensions')
                dim_sizes.append(max(v+1,2))
            self.setup(dim_sizes)
        q_dim = query_dim if query_dim is not None else self.query_dim_idx
        # check known dims
        known_dims = [i for i, v in enumerate(stimulus) if v>= 0 and i != q_dim]
        target_val = stimulus[q_dim]
        #construct vector
        I_pos = self.encode(stimulus)
        full_stimulus = list(stimulus)
        if target_val >= 0:
            full_I_pos = self.encode(full_stimulus)
        else:
            full_I_pos = I_pos.copy()
            
        ## bootstrapper
        recruited=False
        if self.n_clusters ==0:
            self.recruit_cluster(full_I_pos)
            recruited = True
        
        activations = np.array([
            self.cluster_activation(I_pos, self.H_pos[j], known_dims)
            for j in range(self.n_clusters)
        ])
        
        H_out = self.cluster_output(activations)
        winner = int(np.argmax(H_out))
        
        if not self.supervised: #eq 11
            if activations[winner]<self.tau:
                self.recruit_cluster(full_I_pos)
                recruited = True
                activations = np.array([
                    self.cluster_activation(I_pos, self.H_pos[j], known_dims)
                    for j in range(self.n_clusters)
                ])

        C_out = self.output_units(H_out, q_dim) #eq 7
        
        if np.all(C_out == 0): #eq 8
            probs = self.response_prob(C_out)
        else:
            probs = self.response_prob(C_out)
        response = int(np.argmax(probs))
        correct = (target_val >= 0)and (response == target_val)
        
        if self.supervised and target_val >= 0: #eq 10
            if not correct and not recruited:
                self.recruit_cluster(full_I_pos)
                recruited = True
                
                activations = np.array([
                    self.cluster_activation(I_pos, self.H_pos[j], known_dims)
                    for j in range(self.n_clusters)
                ])
                H_out = self.cluster_output(activations)
                winner = int(np.argmax(H_out))
                C_out = self.output__units(H_out, q_dim)
                if np.all(C_out ==0):
                    probs = np.ones(self.dims_sizes[q_dim]) / self.dim_sizes[q_dim]
                else:
                    probs = self.response_prob(C_out)
                response = int(np.argmax(probs))
        
        if target_val >= 0:
            
            t = self.humble_teach(C_out, full_I_pos, q_dim)
            
            self.update_weights(winner, H_out[winner], C_out, t, q_dim)
            
            self.update_position(winner, full_I_pos)
            
            all_dimes = list(range(self.n_dims))
            self.update_tuning(winner, full_I_pos, all_dims)
            
        return{'response': response,
            'prob': probs,
            'correct': correct,
            'n_clusters': self.n_clusters,
            'winner': winner,
            'recruited': recruited,
            'activations': activations
            }
        
    def predict(self, stimulus: list[int], queried_dim: Optional[int] = None) -> np.ndarray:
        
        q_dim = query_dim if query_dim is not None else self.query_dim_idx
        known_dims = [i for i, v in enumerate(stimulus) if v >=0 and i != q-dim]
        I_pos = self.encode_stimulus(stimulus)
        
        if self.n_clusters == 0:
            return np.ones(self.dim_sizes[q_dim]) / self.dim_sizes[q_dim]
        
        activations = np.array([
            self.cluster_activation(I_pos, self.H_pos[j], known_dims)
            for j in range(self.n_clusters)
        ])
        H_out = self.cluster_output(activations)
        C_out=self.output_units(H_out, q_dim)
        
        if np.all(C_out ==0):
            return np.ones(self.dim_sizes[q_dim])/self.dim_sizes[q_dim]
        return self.response_prob(C_out)