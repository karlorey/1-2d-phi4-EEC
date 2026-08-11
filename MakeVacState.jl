using PEPSKit, TensorKit

### Model Parameters ###
L = parse(Int, ARGS[1]) #Length/width of unit cell
n_0 = round(Int, ((L-1)/2))+1 #Index of the point at the center of the lattice (1-based indexing).
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
H, φ, φ2, φ4, Π, Π2 = phi4_model(L, m2, m0, l, Dim, d, a)

### Ground state optimization parameters ###
boundary_alg = (; tol = parse(Float64, ARGS[8]), trunc = (; alg = :FixedSpaceTruncation));
optimizer_alg = (; alg = :LBFGS, tol = parse(Float64, ARGS[9]), maxiter = parse(Int, ARGS[10]), lbfgs_memory = 20);
reuse_env = true
verbosity = 3;

### Initialize PEPS and CTMRG environment
using JLD2
if parse(Bool, ARGS[11]) #Load previous peps and env
    Dim_old = parse(Int, get(ARGS, 12, "$Dim"))
    Dbond_old = parse(Int, get(ARGS, 13, "$Dbond"))
    χ_old = parse(Int, get(ARGS, 14, "$χ"))
    peps₀ = load_object("VacStates/PEPS,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim_old,D=$Dbond_old,chi=$χ_old.jld2")
    env₀ = load_object("VacStates/env,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim_old,D=$Dbond_old,chi=$χ_old.jld2")

    #Changing Dim → Embed the old physical space (codomain) into the new physical space
    if Dim != Dim_old
        M = isometry(ℂ^Dim, ℂ^Dim_old) #Isometry from old Hilbert space to new Hilbert space
        tmap = M * peps₀.A[1,1] #TensorMap embedding previous peps into new Hilbert space (Assuming translational invariance)
        global peps₀ = InfinitePEPS(tmap)
    end

    #Changing Dbond → Embed the old virtual space (domain) into the new virtual space
    if Dbond != Dbond_old
        M1 = isometry(ℂ^Dbond, ℂ^Dbond_old) #Isometry from old virtual space to new virtual space
        M2 = isometry((ℂ^Dbond)', (ℂ^Dbond_old)') #Isometry from old virtual dual space to new virtual dual space
        tmap = peps₀.A[1,1] * (M1' ⊗ M1' ⊗ M2' ⊗ M2') #TensorMap embedding previous peps into new virtual space (Assuming translational invariance)
        global peps₀ = InfinitePEPS(tmap) #Update iPEPS

        env_old = deepcopy(env₀) #Copy the old environment to a new object
        global env₀ = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χ) #Initialize new env structure
        for r in 1:L, c in 1:L #For each site in the unit cell
            for dir in 1:4
                setcorner!(env₀, corner(env_old, dir, 1, 1), dir, r, c) #Set the new corner tensors to the old corner tensors
            end

            for dir in 1:4
                if dir <= 2
                    IsoMap = (id(ℂ^χ) ⊗ M1 ⊗ M2) #Map from old environment codomain to new codomain for N and E tensors
                else
                    IsoMap = (id(ℂ^χ) ⊗ M2 ⊗ M1) #Map from old environment codomain to new codomain for S and W tensors
                end
                edge_new = IsoMap * edge(env_old, dir, 1, 1)
                setedge!(env₀, edge_new, dir, r, c) #Update the edge tensors
            end
        end
    end

    #Changing χ → Embed the old environment into the new environment
    if χ != χ_old
        env_old = deepcopy(env₀) #Copy the old environment to a new object
        global env₀ = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χ) #Initialize new env structure
        M = isometry(ℂ^χ, ℂ^χ_old) #Isometry from old environment space to new environment space
        for r in 1:L, c in 1:L #For each site in the unit cell
            for dir in 1:4
                corner_new = M * corner(env_old, dir, 1, 1) * M' #Map from ℂ^χ_old → ℂ^χ_old to ℂ^χ → ℂ^χ
                setcorner!(env₀, corner_new, dir, r, c) #Update the corner tensors
            end

            for dir in 1:4
                if dir <= 2
                    IsoMap = (M ⊗ id(ℂ^Dbond) ⊗ id((ℂ^Dbond)')) #Map from old environment codomain to new codomain for N and E tensors
                else
                    IsoMap = (M ⊗ id((ℂ^Dbond)') ⊗ id(ℂ^Dbond)) #Map from old environment codomain to new codomain for S and W tensors
                end
                edge_new = IsoMap * edge(env_old, dir, 1, 1) * M'
                setedge!(env₀, edge_new, dir, r, c) #Update the edge tensors
            end
        end
    end

else #Initialize random peps and env
    peps₀ = InfinitePEPS(randn, ComplexF64, ℂ^Dim, ℂ^Dbond; unitcell=(L, L))
    env_random = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χ);
    env₀, info_ctmrg = leading_boundary(env_random, peps₀; boundary_alg...)
end

### Find the ground state ∣Ω⟩ ###
peps_gs, env_gs, E, info_opt = fixedpoint(H, peps₀, env₀; boundary_alg, optimizer_alg, reuse_env, verbosity)

### Save PEPS and CTMRG environment ###
save_object("VacStates/PEPS,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2", peps_gs)
save_object("VacStates/env,L=$L,m2=$m2,l=$l,a=$a,dim=$Dim,D=$Dbond,chi=$χ.jld2", env_gs)