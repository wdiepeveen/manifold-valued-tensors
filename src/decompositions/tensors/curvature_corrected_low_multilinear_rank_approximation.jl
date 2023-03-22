include("naive_low_multilinear_rank_approximation.jl")
include("../../functions/jacobi_field/beta.jl")

using Manifolds, Manopt

function curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank)
    n = size(X)
    d = manifold_dimension(M)
    r = Tuple(min.(n, rank))
    dims = length(n)
    
    R_q, U  = naive_low_multilinear_rank_approximation(M, q, X, r)

    # construct linear system
    J = CartesianIndices(n)
    L = CartesianIndices(r)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n

    A = zeros(d, r..., d, r...)
    b = zeros(d, r...)
    for jᵢ in J
        ONBⱼᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[jᵢ]))
        Θⱼᵢ = ONBⱼᵢ.data.vectors
        κⱼᵢ = ONBⱼᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κⱼᵢ .*= distance(M, q, X[jᵢ])^2
        end

        A += [sum([β(κⱼᵢ[j])^2 * inner(M, q, get_vector(M, q, (prod([U[i][jᵢ[i],l₁[i]] for i=1:dims]) .* Matrix(I, d, d))[:,k₁], DefaultOrthonormalBasis()), Θⱼᵢ[j]) * inner(M, q, get_vector(M, q, (prod([U[i][jᵢ[i],l₂[i]] for i=1:dims]) .* Matrix(I, d, d))[:,k₂], DefaultOrthonormalBasis()), Θⱼᵢ[j])  for j=1:d]) for k₁=1:d, l₁ in L, k₂=1:d, l₂ in L]
        b += [sum([β(κⱼᵢ[j])^2 * inner(M, q, get_vector(M, q, (prod([U[i][jᵢ[i],l₁[i]] for i=1:dims]) .* Matrix(I, d, d))[:,k₁], DefaultOrthonormalBasis()), Θⱼᵢ[j]) * inner(M, q, log_q_X[jᵢ], Θⱼᵢ[j])  for j=1:d]) for k₁=1:d, l₁ in L]
    end

    # reshape
    AA = reshape(A, (d * prod(r), d * prod(r)))
    bb = reshape(b, (d * prod(r)))

    # solve linear system
    VVₖₗ = AA\bb
    Vₖₗ = reshape(VVₖₗ, (d, r...))

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q),[Vₖₗ[:,l] for l in L], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U

end
