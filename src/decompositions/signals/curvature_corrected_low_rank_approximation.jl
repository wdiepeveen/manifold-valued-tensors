include("naive_low_rank_approximation.jl")
include("../../functions/jacobi_field/beta.jl")

using Manifolds, Manopt

function curvature_corrected_low_rank_approximation(M, q, X, rank)
    n = size(X)
    d = manifold_dimension(M)
    r = min(n[1], d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r) 
    # construct linear system
    J = CartesianIndices(n)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n

    A = zeros(d, r, d, r)
    b = zeros(d, r)
    for j₁ in J
        ONBⱼ₁ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[j₁]))
        Θⱼ₁ = ONBⱼ₁.data.vectors
        κⱼ₁ = ONBⱼ₁.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κⱼ₁ .*= distance(M, q, X[j₁])^2
        end

        A += [sum([β(κⱼ₁[j])^2 * inner(M, q, get_vector(M, q, (U[j₁,l₁] .* Matrix(I, d, d))[:,k₁], DefaultOrthonormalBasis()), Θⱼ₁[j]) * inner(M, q, get_vector(M, q, (U[j₁,l₂] .* Matrix(I, d, d))[:,k₂], DefaultOrthonormalBasis()), Θⱼ₁[j])  for j=1:d]) for k₁=1:d, l₁=1:r, k₂=1:d, l₂=1:r]
        b += [sum([β(κⱼ₁[j])^2 * inner(M, q, get_vector(M, q, (U[j₁,l₁] .* Matrix(I, d, d))[:,k₁], DefaultOrthonormalBasis()), Θⱼ₁[j]) * inner(M, q, log_q_X[j₁], Θⱼ₁[j])  for j=1:d]) for k₁=1:d, l₁=1:r]
    end

    # add regularisation
    AA = reshape(A, (d * r, d * r))
    bb = reshape(b, (d * r))

    # solve linear system
    VVₖₗ = AA\bb
    Vₖₗ = reshape(VVₖₗ, (d, r))

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q),[Vₖₗ[:,l] for l=1:r], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U
end
