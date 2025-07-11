# SHASTRY SUTHERLAND MODEL 
### Project for the exam "Introduzione ai sistemi quantistici a molti corpi"
Notebooks for the diagonalization of the Shastry-Sutherland model 4x4:
$H = J1\sum_{<i,j>} S_iS_j +  J2\sum_{<<i,j>>} S_iS_j$
where the first term runs over the nearest neighbours and the second term runs over some selected diagonal bonds, called dimers.

* "Hamiltonian_definition" contains all the functions used to define the lattice, the hamiltonian, the correlation function...
* "exact diagonalization 4x4 Shastry Sutherland" is the main notebook, it contains the exact diagonalization of the hamiltonians, and the analysis of the energy levels, of the energy gap and the correlations changing J1 and J2.
  
* "2x2 Shastry Sutherland model" contains some preliminary tests, made on a smaller lattice.
* "4x4 Shastry Sutherland model" contains the diagonalization using Lanczos method.
* "comparison" contains a brief comparison between exact diagonalization and Lanczos results.
