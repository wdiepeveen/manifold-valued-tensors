function exact_loss(M::AbstractManifold, q, X, U, V)
    n = size(X)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    I = CartesianIndices(n)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * transpose(V))[i,:] for i in I], Ref(DefaultOrthonormalBasis()))

    exp_q_Ξ =   exp.(Ref(M), Ref(q), Ξ)
    return sum(distance.(Ref(M), X, exp_q_Ξ).^2) / ref_distance
end

function exact_loss(M::AbstractManifold, q, X, Ξ)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)

    exp_q_Ξ =   exp.(Ref(M), Ref(q), Ξ)
    return sum(distance.(Ref(M), X, exp_q_Ξ).^2) / ref_distance
end