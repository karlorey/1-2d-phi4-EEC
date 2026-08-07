using PEPSKit, TensorKit

### Model Parameters ###
N = parse(Int, ARGS[1]) #Number of sites in unit cell
n_0 = round(Int, ((N-1)/2))+1 #Index of the point at the center of the lattice (1-based indexing).
m2 = parse(Float64, ARGS[2]) #Bare mass (squared)
m0 = 1 #Basis frequency
l = parse(Float64, ARGS[3]) #phi^4 coupling strength
Dim = parse(Int, ARGS[8]) #Truncated local Hilbert space dimension
V = ComplexSpace(Dim) #Truncated Hilbert space (ℂ^D)
d = 2 #Number of spatial dimensions
a = parse(Float64, ARGS[7]) #Lattice spacing

### iPEPS Dimensions ###
Dbond = parse(Int, ARGS[9])
χ = parse(Int, ARGS[10])

### phi4 Hamiltonian ###
include("phi4_Hamiltonian.jl")
H, φ, φ2, φ4, Π, Π2 = phi4_model(N, m2, m0, l, Dim, d, a)

### Load ground state PEPS and CTMRG environment ###
using JLD2
peps_gs = load_object("VacStates/PEPS,N=$N,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")
env_gs = load_object("VacStates/env,N=$N,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")

### Apply φ(0) to ∣Ω⟩ ###
Ω = deepcopy(peps_gs)
Ω.A[n_0, n_0] = φ * Ω.A[n_0, n_0]

### Define T₀ᵢ(x) and H(0) ###
T0i_TMap = 1/2 * (φ ⊗ Π - Π ⊗ φ) #Energy flux TensorMap
h_site = a^d * ((1/2 * Π2) + (1/a^2 * φ2) + (1/a^2 * φ2) + (1/2 * m2 * φ2) + (l/24 * φ4)) #Onsite Hamiltonian TensorMap
h_bond = -1/2 * a^d * 1/a^2 * φ ⊗ φ #Nearest-neighbor interaction Hamiltonian TensorMap (split into four directions)

lattice = fill(V, N, N)
T0i = LocalOperator(lattice,
    [(n_0, n_0), (n_0+1, n_0)] => T0i_TMap,
    [(n_0, n_0), (n_0-1, n_0)] => T0i_TMap,
    [(n_0, n_0), (n_0, n_0+1)] => T0i_TMap,
    [(n_0, n_0), (n_0, n_0-1)] => T0i_TMap)
    
H0 = LocalOperator(lattice,
    [(n_0, n_0)] => h_site,
    [(n_0, n_0), (n_0+1, n_0)] => h_bond,
    [(n_0, n_0), (n_0-1, n_0)] => h_bond,
    [(n_0, n_0), (n_0, n_0+1)] => h_bond,
    [(n_0, n_0), (n_0, n_0-1)] => h_bond)

### Time evolution parameters ###
ti = parse(Float64, ARGS[4]) #Initial time
tf = parse(Float64, ARGS[5]) #Final time
Δt = parse(Float64, ARGS[6]) #Time step size
δt = 0.01 #Trotter step size
nsteps = round(Int, Δt/δt) #Number of steps per trotter evolution
t_range = range(ti, tf, step=Δt) #Time range

### Initialize time evolution (real-time SimpleUpdate) ###
trunc_peps = truncerror(; atol = 1e-10) & truncrank(Dbond)
alg = SimpleUpdate(; trunc = trunc_peps, imaginary_time = false) #Time evolution algorithm
boundary_alg = (; tol = 1e-8, trunc = (; alg = :FixedSpaceTruncation)) #CTMRG boundary algorithm

ψ = Ω
env = env_gs
wts = SUWeight(Ω)

Edens0 = ComplexF64[]
Eflux = ComplexF64[]

### Perform time evolution ###
for i in 1:length(t_range)
    if i == 1 #t=0
        evH = expectation_value(Ω, H0, env_gs)
        evT0i = expectation_value(Ω, T0i, env_gs)
    else #t>0
        global ψ, env, wts
        ψ, wts = time_evolve(ψ, H, δt, nsteps, alg, wts)
        env, info_ctmrg = leading_boundary(env, ψ; boundary_alg...)

        evH = expectation_value(ψ, H0, env)
        evT0i = expectation_value(ψ, T0i, env)
    end

    push!(Edens0, evH)
    push!(Eflux, evT0i)
end

### Save Data ###
using HDF5
data = h5open("ECons_data/EVs,N=$N,m2=$m2,l=$l,ti=$ti,tf=$tf,dt=$Δt,a=$a,dim=$Dim,D=$Dbond,chi=$χ.h5", "w")
data["Edens0"] = Edens0
data["Eflux"] = Eflux
close(data)