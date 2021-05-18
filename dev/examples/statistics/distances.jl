using Distances, LazyDistances
using LinearAlgebra
using Random
Random.seed!(0)

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 3);

dist = pairwise(Euclidean(), X, Y)

lazy_dist = let X=X, Y=Y
    # defines:
    #   ĩ := σ(i)
    #   j̃ := σ(j)
    # where `i` and `j` are `CartesianIndex`.
    # We don't need to operate on index `i` and `j` here, so use `identity`.
    index_map = (
        identity,
        identity
    )
    # defines:
    #   p = A[ĩ] := @view X[:, ĩ]
    #   q = B[j̃] := @view Y[:, j̃]
    # Here the input A is axes(X, 2) and is not used.
    getindex_op = (
        (ax, i) -> view(X, :, i),
        (ax, j) -> view(Y, :, j)
    )
    # defines:
    #   out[i, j] := d(p, q)
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

