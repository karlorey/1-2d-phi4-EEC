using PEPSKit, TensorKit

function phi4_model(N, m2, m0, l, Dim, d, a)
    ### Field Operators ###
    function a_matrix(D)
        A = zeros(D, D)
        for n in 1:D-1
            A[n, n+1] = sqrt(n)
        end
        return A
    end

    #Field operator matrices
    phi_matrix = D -> (a_matrix(D) + a_matrix(D)')/sqrt(2*m0)
    phi2_matrix = D -> phi_matrix(D)^2
    phi4_matrix = D -> phi_matrix(D)^4
    pi_matrix = D -> im*sqrt(m0/2)*(a_matrix(D)' - a_matrix(D))
    pi2_matrix = D -> -(m0 / 2) * ((a_matrix(D)' - a_matrix(D)) * (a_matrix(D)' - a_matrix(D)))

    #Truncated Hilbert space (ℂ^D)
    V = ComplexSpace(Dim)

    #Convert matrices to TensorMap
    φ = TensorMap(phi_matrix(Dim), V ← V)
    φ2 = TensorMap(phi2_matrix(Dim), V ← V)
    φ4 = TensorMap(phi4_matrix(Dim), V ← V)
    Π = TensorMap(pi_matrix(Dim), V ← V)
    Π2 = TensorMap(pi2_matrix(Dim), V ← V)

    ### phi4 Hamiltonian ###
    function phi4_H(lattice::InfiniteSquare; m2=m2, l=l, a=a, d=d)
        h_site = a^d * ((1/2 * Π2) + (1/a^2 * φ2) + (1/a^2 * φ2) + (1/2 * m2 * φ2) + (l/24 * φ4)) #Onsite Hamiltonian
        h_bond = -1/a^2 * a^d * φ ⊗ φ #Nearest-neighbor interaction Hamiltonian
        spaces = fill(V, (lattice.Nrows, lattice.Ncols))

        return LocalOperator(
            spaces,
            (neighbor => h_bond for neighbor in nearest_neighbours(lattice))...,
            ([idx] => h_site for idx in vertices(lattice))...,
        )
    end

    H = phi4_H(InfiniteSquare(N, N))
    return H, φ, φ2, φ4, Π, Π2
end