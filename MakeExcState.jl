using PEPSKit, TensorKit

### Model Parameters ###
L = parse(Int, ARGS[1]) #Length/width of unit cell
n_0 = round(Int, ((L-1)/2))+1 #Index of the point at the center of the lattice (1-based indexing).
m2 = parse(Float64, ARGS[2]) #Bare mass (squared)
m0 = 1 #Basis frequency
l = parse(Float64, ARGS[3]) #phi^4 coupling strength
Dim = parse(Int, ARGS[5]) #Truncated local Hilbert space dimension
V = ComplexSpace(Dim) #Truncated Hilbert space (ℂ^D)
d = 2 #Number of spatial dimensions
a = parse(Float64, ARGS[4]) #Lattice spacing

### iPEPS Dimensions ###
Dbond = parse(Int, ARGS[6])
χ = parse(Int, ARGS[7])

### phi4 Hamiltonian ###
include("phi4_Hamiltonian.jl")
H, φ, φ2, φ4, Π, Π2 = phi4_model(L, m2, m0, l, Dim, d, a)

### Load the 1x1 ground state PEPS and CTMRG environment ###
using JLD2
peps_1 = load_object("VacStates/PEPS,L=1,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")
env_1 = load_object("VacStates/env,L=1,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")

#Convert the 1x1 vacuum tensor to LxL
A1 = peps_1.A[1,1] #1x1 vacuum tensor
AL = fill(A1, (L,L)) #Vacuum tensor copied over an LxL unit cell
peps_gs = InfinitePEPS(AL)

#Convert the 1x1 environment to LxL
env_gs = CTMRGEnv(randn, ComplexF64, peps_gs, ℂ^χ); #Generate structure of LxL environment

#Replace corner and edge tensors of LxL environment with the 1x1 corner and edge tensors
for r in 1:L, c in 1:L
    for dir in 1:4
        setcorner!(env_gs, corner(env_1, dir, 1, 1), dir, r, c)
    end

    for dir in 1:4
        setedge!(env_gs, edge(env_1, dir, 1, 1), dir, r, c)
    end
end

### Save the LxL ground state PEPS and CTMRG environment ###
jldsave("VacStates/Vac,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2"; peps_gs, env_gs)

### Apply φ(0) to ∣Ω⟩ ###
Ω = deepcopy(peps_gs)
Ω.A[n_0, n_0] = φ * Ω.A[n_0, n_0]

### Construct PEPS and env from φ(0)∣Ω⟩ ###
boundary_alg = (; tol = 1e-8, trunc = (; alg = :FixedSpaceTruncation)) #CTMRG boundary algorithm
println("Constructing φ(0)∣Ω⟩ iPEPS, environment, and weights...")
flush(stdout)
ψ = Ω #Initialize the iPEPS ψ as the excited vacuum state iPEPS Ω
env, info_ctmrg = leading_boundary(env_gs, ψ; boundary_alg...) #Initialize the env with ψ
flush(stderr)
wts = SUWeight(ψ) #Initialize the simple update weights with ψ
println("Done.")
flush(stdout)

### Save PEPS and CTMRG environment ###
jldsave("ExcStates/Exc,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2"; ψ, env, wts)