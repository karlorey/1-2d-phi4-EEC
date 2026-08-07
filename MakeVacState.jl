using PEPSKit, TensorKit

### Model Parameters ###
N = parse(Int, ARGS[1]) #Number of sites in unit cell
n_0 = round(Int, ((N-1)/2))+1 #Index of the point at the center of the lattice (1-based indexing).
m2 = parse(Float64, ARGS[2]) #Bare mass (squared)
m0 = 1 #Basis frequency
l = parse(Float64, ARGS[3]) #phi^4 coupling strength
Dim = parse(Int, ARGS[5]) #Truncated local Hilbert space dimension
d = 2 #Number of spatial dimensions
a = parse(Float64, ARGS[4]) #Lattice spacing

### iPEPS Dimensions ###
Dbond = parse(Int, ARGS[6])
χ = parse(Int, ARGS[7])

### phi4 Hamiltonian ###
include("phi4_Hamiltonian.jl")
H, φ, φ2, φ4, Π, Π2 = phi4_model(N, m2, m0, l, Dim, d, a)

### Ground state optimization parameters ###
boundary_alg = (; tol = 1e-8, trunc = (; alg = :FixedSpaceTruncation));
optimizer_alg = (; alg = :LBFGS, tol = 1e-6, maxiter = parse(Int, ARGS[8]), lbfgs_memory = 16);
reuse_env = true
verbosity = 3;

### Initialize PEPS and CTMRG environment
peps₀ = InfinitePEPS(randn, ComplexF64, ℂ^Dim, ℂ^Dbond; unitcell=(N, N))
env_random = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χ);
env₀, info_ctmrg = leading_boundary(env_random, peps₀; boundary_alg...)

### Find the ground state ∣Ω⟩ ###
peps_gs, env_gs, E, info_opt = fixedpoint(H, peps₀, env₀; boundary_alg, optimizer_alg, reuse_env, verbosity)

### Save PEPS and CTMRG environment ###
using JLD2
save_object("VacStates/PEPS,N=$N,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2", peps_gs)
save_object("VacStates/env,N=$N,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2", env_gs)