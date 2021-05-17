using Distances, LazyDistances
using LinearAlgebra
using Random
Random.seed!(0)

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 3);

dist = pairwise(Euclidean(), X, Y)

lazy_dist = let X=X, Y=Y

    index_map = (
        identity,
        identity
    )

    getindex_op = (
        (ax, i) -> view(X, :, i),
        (ax, j) -> view(Y, :, j)
    )

    d = Euclidean()
    PairwiseDistance(index_map, getindex_op, d, (axes(X, 2), axes(Y, 2)))
end

dist == lazy_dist

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 2);
dist = colwise(Euclidean(), X, Y)

lazy_dist = let X=X, Y=Y
    getindex_op = (
        (ax, i) -> view(X, :, i),
        (ax, j) -> view(Y, :, j)
    )
    PairwiseDistance(identity, getindex_op, Euclidean(), (axes(X, 2), axes(Y, 2)))
end

diag(lazy_dist) == dist

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

