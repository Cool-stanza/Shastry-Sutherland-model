import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import block_diag
from scipy.sparse.linalg import eigsh
import scipy.linalg as la


#coordinates of lattice points
def generate_lattice(Lx,Ly):
    coor = []                     
    for i in range(Lx):       
         for j in range(Ly):   
            coor.append((i,j))
    return coor

# NEAREST NEIGHBORS
def nearest_neighbors(coor,Lx,Ly):
    neighbors = []
    for idx, (i, j) in enumerate(coor):
        neighbor_list = []
    
        right = (i+1)%Lx  # if i==Lx-1 nn==0 
        neighbor_list.append((right, j))  
    
        top = (j-1)%Ly  # if j==0, nn==Ly-1 (==3)
        neighbor_list.append((i, top))  
    
        neighbors.append(neighbor_list)

    return neighbors

def index_nn(coor,Lx,Ly):
    neighbors_indices = []
    for neighbor_list in nearest_neighbors(coor,Lx,Ly):
        indices = [coor.index(v) for v in neighbor_list]  # find the index of each nn
        neighbors_indices.append(indices)    
    return neighbors_indices


# NEXT NEIGHBORS - DIAGONALS
def next_neighbors(coor,Lx,Ly):
    avoid = []
    diagonals = [None]*Lx*Ly
    for idx, (i, j) in enumerate(coor):   
        if i%2==0 and j%2==0:
            next_x = (i+1)  
            next_y = (j+1)
            diagonals[idx]=((next_x, next_y))
            avoid.append((i,j))
        
        if i%2!=0 and j%2!=0:     
            next_x = (i-1)  
            next_y = (j-1)
            diagonals[idx]=((next_x, next_y))
            avoid.append((i,j))  

    for idx, (i, j) in enumerate(coor):  
        if (i,j) in avoid:
            continue
        elif i%2==0:
            next_x = (i-1)%Lx 
            next_y = (j+1)%Ly 
            #print(i,j, "-->",next_x, next_y )
            diagonals[idx]=((next_x,next_y))
        elif i%2!=0:
            next_x = (i+1)%Lx 
            next_y = (j-1)%Ly 
            #print(i,j, "-->",next_x, next_y )
            diagonals[idx]=((next_x,next_y))
    return diagonals


def index_nnn(coor,Lx,Ly):
    diag_indices = []
    for diag_point in next_neighbors(coor,Lx,Ly):
        if diag_point is not None:  #
            index = coor.index(diag_point)  # Trova l'indice della tupla in coor
            diag_indices.append([index])  # Crea una lista con l'indice trovato
        else:
            diag_indices.append([])     
    return diag_indices


#PLOTTING LATTICE
def plot_lattice(coor, neighbors_indices, diag_indices):
    x = [p[0] for p in coor]
    y = [p[1] for p in coor]


    plt.figure(figsize=(3,3))
    plt.scatter(x, y)
    for idx, point in enumerate(coor):
         plt.text(point[0] + 0.05, point[1] + 0.05, f"{idx}", fontsize=10)

    for i, diag_list in enumerate(diag_indices):
        for diag_idx in diag_list:
            x_vals = [coor[i][0], coor[diag_idx][0]]
            y_vals = [coor[i][1], coor[diag_idx][1]]
            plt.plot(x_vals, y_vals, 'k--', alpha=0.5)  # nn line
        
    for i, neighbor_list in enumerate(neighbors_indices):
        for neighbor_idx in neighbor_list:
            x_vals = [coor[i][0], coor[neighbor_idx][0]]
            y_vals = [coor[i][1], coor[neighbor_idx][1]]
            plt.plot(x_vals, y_vals, 'k-', alpha=0.5)  # nn line
    plt.show()

    #check
    for idx, (i,j) in enumerate(coor):
        print(f"{idx} --> Nearest neighbors: {neighbors_indices[idx]}", f" -- Next neighbor: {diag_indices[idx]}")

#-------------------------------------------------------------------------------------------------------------------------

# HAMILTONIAN
def flip(state,i,j): #flippa lo spin degli indici i e j
    return state ^ (2**i + 2**j) # ^==xor

def Hamiltonian(J1, J2, state, Lx, Ly, neighbors_indices, diag_indices):
    result = []
    seen_states_1 = set()  # Per evitare duplicati
    seen_states_2 = set() 
    seen_states_d = set() 
    
    coeff_nn = 0
    coeff_nnn = 0
    coeff_field = 0
    for i in range(Lx*Ly):
        n = (state & 2**i)/2**i #extract bit at position i --> 0 or 1 --> poi (2n-1)/2 per avere -1/2, +1/2
        
        nn_x = neighbors_indices[i][0] #nn dx
        nn_y = neighbors_indices[i][1] #nn up
        nn_d = diag_indices[i][0] #nnn

        nn_1 = (state & 2**nn_x)/2**nn_x
        nn_2 = (state & 2**nn_y)/2**nn_y
        coeff_nn += (2*n-1)*(2*nn_1-1)/4 + (2*n-1)*(2*nn_2-1)/4 # sarebbe (2*n-1)/2 * (2*nn-1)/2 
        if n!=nn_1 : 
            new_state_1 = flip(state,i,nn_x)
            if new_state_1 not in seen_states_1:
                result.append([J1/2,new_state_1])
                seen_states_1.add(new_state_1)
        if n!=nn_2 : 
            new_state_2 = flip(state,i,nn_y)
            #if new_state_2 not in seen_states_2:
            result.append([J1/2,new_state_2])
             #   seen_states_2.add(new_state_2)
        
        nnn = (state & 2**nn_d)/2**nn_d
        coeff_nnn += (2*n-1)*(2*nnn-1)/8 #1/2 in più perchè i vicini in diagonale li sto contando due volte
        if n!=nnn : 
            new_state_d = flip(state,i,nn_d)
            if new_state_d not in seen_states_d:
                result.append([J2/2,new_state_d])
                seen_states_d.add(new_state_d)
          
    result.append([J1*coeff_nn+J2*coeff_nnn, state])
    return result 


#MATRICE DIAGONALE A BLOCCHI
def build_basisN(L,N): #L=Lx*Ly number of sites, N number of spin up in the sector
    basisN = []
    for n in range(2**L):
        particle_count = bin(n).count('1') #count the 1 in n
        if particle_count == N:
            basisN.append(n) #check if n belongs to the N sector
    return basisN


def build_HN(Lx,Ly,N,J1,J2,neighbors_indices,diag_indices):
    L=Lx*Ly
    basisN = build_basisN(L,N)
    dimN = len(basisN) #dimension of the subspace S_N
    HN = dok_matrix((dimN,dimN)) #crea matrice vuota sparsa

    for b,n in enumerate(basisN): #b index, n binary state
        output = Hamiltonian(J1,J2,n,Lx,Ly,neighbors_indices,diag_indices) #H|n>
        for coeff,m in output:
            try:
                a = basisN.index(m)
                HN[a,b]+=coeff
            except ValueError:
                continue
    return HN.tocsr() #csr


def H_diag_block(Lx,Ly,J1,J2,neighbors_indices,diag_indices):
    L=Lx*Ly
    blocks = []
    for N in range(L+1):
        basisN = build_basisN(L,N)
        if len(basisN) > 0:
            HN = build_HN(Lx,Ly,N,J1,J2,neighbors_indices,diag_indices)
            blocks.append(HN)
    H = block_diag(blocks, format="csr")
    return H


#---------------------------------------------------------------------------------------------
#CORRELATIONS 1
def build_SiSj(Lx,Ly,N,neighbors_indices,diag_indices):
    L = Lx*Ly
    basisN = build_basisN(L,N)
    dimN = len(basisN)
    state_index = {state: idx for idx, state in enumerate(basisN)}

    SzSz_nn = dok_matrix((dimN,dimN))
    SzSz_nnn = dok_matrix((dimN,dimN))

    Sflip_nn = dok_matrix((dimN,dimN))
    Sflip_nnn = dok_matrix((dimN,dimN))

    for b,state in enumerate(basisN): # b index, state binary state
        for i in range(L):
            ni = (state & 2**i)/2**i #extract bit at position i --> 0 or 1
            
            for j in neighbors_indices[i]:
                nj = (state & 2**j)/2**j
                #SzSz
                SzSz_nn[b,b] += (2*ni-1)*(2*nj-1)/4
                #S+S-
                if ni==0 and nj==1:       
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        b_new = state_index[new_state]
                        Sflip_nn[b_new,b] += 1/2
                #S-S+
                if ni==1 and nj==0:       
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        b_new = state_index[new_state]
                        Sflip_nn[b_new,b] += 1/2


            for j in diag_indices[i]:
                nj = (state & 2**j)/2**j
                #SzSz
                SzSz_nnn[b,b] += (2*ni-1)*(2*nj-1)/4
                #S+S-
                if ni==0 and nj==1:       
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        b_new = state_index[new_state]
                        Sflip_nn[b_new,b] += 1/2
                #S-S+
                if ni==1 and nj==0:       
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        b_new = state_index[new_state]
                        Sflip_nn[b_new,b] += 1/2

    return SzSz_nn.tocsr(), SzSz_nnn.tocsr()

def spin_corr(Lx,Ly,psi1,psi2,N,neighbors_indices,diag_indices):
    L=Lx*Ly
    SiSj_nn, SiSj_nnn = build_SiSj(Lx,Ly,N,neighbors_indices,diag_indices)

    SiSj_nn_exval = np.vdot(psi1, SiSj_nn @ psi2)
    SiSj_nnn_exval = np.vdot(psi1, SiSj_nnn @ psi2)

    total_nn_links = sum(len(neighbors_indices[i]) for i in range(L)) 
    total_nnn_links = sum(len(diag_indices[i]) for i in range(L)) 
    #print(total_nn_links)
    #print(total_nnn_links)

    SiSj_nn_exval = SiSj_nn_exval / total_nn_links
    SiSj_nnn_exval = SiSj_nnn_exval / total_nnn_links

    return SiSj_nn_exval, SiSj_nnn_exval
            
#CORRELATIONS 2
def sz(state, site):
    n = (state & 2**site)/2**site
    sz = (2.*n-1)/2.
    return sz

def compute_sisj_correlations(GS, N, L):
    basisN = build_basisN(L,N)
    state_index = {state: idx for idx, state in enumerate(basisN)}
    correlations = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            total = 0.0
            tot_szsz = 0.0
            tot_1 = 0.0
            tot_2 = 0.0
            for k, state in enumerate(basisN):
                amp = GS[k]   #per ogni stato della base prendo la relativa ampiezza nel GS
                #SzSz
                sz_i = sz(state, i)
                sz_j = sz(state, j)
                tot_szsz += (amp**2) * sz_i * sz_j
                
                ni = (state & 2**i)/2**i
                nj = (state & 2**j)/2**j
                #S+S-
                if ni==0 and nj==1:
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        new_k = state_index[new_state]
                        new_amp = GS[new_k]
                        tot_1 += amp*new_amp #ampiezze reali
                if i==j:           
                    tot_1 += 1/2 * amp**2 #1/2 perchè sto contando sia i=j sia j=i
                #S-S+
                if ni==1 and nj==0:
                    new_state = flip(state,i,j)
                    if new_state in state_index:
                        new_k = state_index[new_state]
                        new_amp = GS[new_k]
                        tot_2 += amp*new_amp #ampiezze reali
                if i==j:
                    tot_2 += 1/2 * amp**2

            # if i==0: #checking
            #     print(f'({i},{j}), szsz={tot_szsz}, s+s-={tot_1}, s-s+={tot_2}')

            total = tot_szsz + 0.5*(tot_1 + tot_2)
            correlations[i, j] = total
    return correlations

def sisj_mean(state,N,L,neighbors_indices,diag_indices):

    corr = compute_sisj_correlations(state, N, L)

    sisj_nn = 0.0
    sisj_nnn = 0.0
    for i in range(L):
        for j in range(L):
            if j in neighbors_indices[i]:
                sisj_nn += corr[i,j]
            if j in diag_indices[i]:
                sisj_nnn += corr[i,j]

    total_nn_links = sum(len(neighbors_indices[i]) for i in range(L)) 
    total_nnn_links = sum(len(diag_indices[i]) for i in range(L)) 
    sisj_nn = sisj_nn/total_nn_links
    sisj_nnn = sisj_nnn/total_nnn_links

    return sisj_nn, sisj_nnn
            


