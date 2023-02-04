include("tangent_space_SVD.jl")

function naive_low_rank_approximation(M, q, X, rank)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # TODO call tangentSVD
    R_q, U = tangent_space_SVD(M, q, log_q_X, rank)
    return R_q, U
end