using Distances, LazyDistances
using LinearAlgebra
using Random
Random.seed!(0)

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 3);

dist = pairwise(Euclidean(), X, Y)

lazy_dist = let X=X, Y=Y
    T = result_type(Euclidean(), X, Y)
    PairwiseDistance{T}(Euclidean(), axes(X, 2), axes(Y, 2)) do i, j
        @views X[:, i...], Y[:, j...]
    end
end

dist == lazy_dist

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 2);
dist = colwise(Euclidean(), X, Y)

lazy_dist = let X=X, Y=Y
    T = result_type(Euclidean(), X, Y)
    eval_op(i, j) = Euclidean()(view(X, :, i...), view(Y, :, j...))
    PairwiseDistance{Float64}(eval_op, axes(X, 2), axes(Y, 2))
end

Diagonal(lazy_dist) == Diagonal(dist)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

