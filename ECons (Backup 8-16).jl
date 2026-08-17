using PEPSKit, TensorKit

### Model Parameters ###
L = parse(Int, ARGS[1]) #Length/width of unit cell
n_0 = round(Int, ((L-1)/2))+1 #Index of the point at the center of the lattice (1-based indexing).
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
H, φ, φ2, φ4, Π, Π2 = phi4_model(L, m2, m0, l, Dim, d, a)

### Load PEPS and CTMRG environment ###
using JLD2
boundary_alg = (; tol = 1e-8, trunc = (; alg = :FixedSpaceTruncation)) #CTMRG boundary algorithm
function LoadState()
    if !parse(Bool, ARGS[12])
        #Load ground state
        peps_gs = load_object("VacStates/PEPS,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")
        env_gs = load_object("VacStates/env,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")

        #Apply φ(0) to ∣Ω⟩
        Ω = deepcopy(peps_gs)
        Ω.A[n_0, n_0] = φ * Ω.A[n_0, n_0]

        #Construct PEPS and env from φ(0)∣Ω⟩
        println("Constructing φ(0)∣Ω⟩ iPEPS, environment, and weights...")
        flush(stdout)
        ψ = Ω #Initialize the iPEPS ψ as the excited vacuum state iPEPS Ω
        env, info_ctmrg = leading_boundary(env_gs, ψ; boundary_alg...) #Initialize the env with ψ
        flush(stderr)
        wts = SUWeight(ψ) #Initialize the simple update weights with ψ
        println("Done.")
        flush(stdout)
    else
        #Load previous state
        ψ, env, wts = load("EvolStates/ECons,L=$L,m2=$m2,l=$l,ti=0.0,tf=$(parse(Float64, ARGS[13])),dt=$Δt,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2")
        println("Loaded state.")
        flush(stdout)
    end

    return ψ, env, wts
end

### Define T₀ᵢ(x) and H(0) ###
T0i_TMap = 1/2 * (φ ⊗ Π - Π ⊗ φ) #Energy flux TensorMap
h_site = a^d * ((1/2 * Π2) + (1/a^2 * φ2) + (1/a^2 * φ2) + (1/2 * m2 * φ2) + (l/24 * φ4)) #Onsite Hamiltonian TensorMap
h_bond = -1/2 * a^d * 1/a^2 * φ ⊗ φ #Nearest-neighbor interaction Hamiltonian TensorMap (split into four directions)

lattice = fill(V, L, L)
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

### Perform time evolution ###
function Evolve()
    ψ, env, wts = LoadState()
    Edens0 = ComplexF64[]
    Eflux = ComplexF64[]

    println("Beginning real-time evolution.")
    flush(stdout)
    for t in t_range
        if t == 0 #t=0
            println("Starting t=0.0. Calculating expectation values...")
            evH = expectation_value(ψ, H0, env)
            evT0i = expectation_value(ψ, T0i, env)
            println("Done.")
            flush(stdout)
        else #t>0
            println("Starting t=$t. Beginning time evolution...")
            flush(stdout)
            ψ, wts = time_evolve(ψ, H, δt, nsteps, alg, wts; symmetrize_gates = parse(Bool, ARGS[11]))
            flush(stderr)
            println("Finished time evolving. Updating environment...")
            flush(stdout)
            env, info_ctmrg = leading_boundary(env, ψ; boundary_alg...)
            flush(stderr)
            println("Finished updating.")
            flush(stdout)

            println("Calculating expectation values...")
            evH = expectation_value(ψ, H0, env)
            evT0i = expectation_value(ψ, T0i, env)
            println("Done.")
            flush(stdout)
        end

        push!(Edens0, evH)
        push!(Eflux, evT0i)
    end

    #Save the state, return the energy density and energy flux
    jldsave("EvolStates/ECons,L=$L,m2=$m2,l=$l,ti=0.0,tf=$tf,dt=$Δt,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2"; ψ, env, wts)
    return Edens0, Eflux
end
Edens0, Eflux = Evolve()

### Save Data ###
using HDF5
data = h5open("ECons_data/EVs,L=$L,m2=$m2,l=$l,ti=$ti,tf=$tf,dt=$Δt,a=$a,dim=$Dim,D=$Dbond,chi=$χ.h5", "w")
data["Edens0"] = Edens0
data["Eflux"] = Eflux
close(data)