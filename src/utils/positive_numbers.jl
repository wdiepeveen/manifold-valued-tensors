using Manifolds, ManifoldsBase

function get_coordinates_orthonormal!(::PositiveNumbers, Xⁱ, p, X, ::ManifoldsBase.RealNumbers)
    Xⁱ = X / p
    return Xⁱ
end