# ---
# title: Row/Column-wise Distances
# author: Johnny Chen
# date: 2021-5-17
# ---

# This example shows how `LazyDistances.PairwiseDistance` can be used to get a better
# `Distances.pairwise` performance.
#
# !!! warning
#     The benchmark result `@btime` is not generated automatically so it may be outdated.
#     If so please file an issue/PR to update it.
#

using Distances, LazyDistances
using LinearAlgebra
using Random
Random.seed!(0)

# ## pairwise distance

X = rand(1:10, 100, 2);
Y = rand(1:10, 100, 3);

dist = pairwise(Euclidean(), X, Y)

# `Distances.pairwise` explicitly generates a large array, which does not perform well for massive
# scale data. [`PairwiseDistance`](@ref) mimics the pairwise operation in a lazy and optionally
# cached manner, so that you don't need to worry about memory allocation and the overhead due to GC.
# Unlike `pairwise`, `PairwiseDistance` does not assume the data to be row or column shaped; instead
# you need to manually specify how data pair is constructed:

lazy_dist = let X=X, Y=Y
    ## defines:
    ##   ĩ := σ(i)
    ##   j̃ := σ(j)
    ## where `i` and `j` are `CartesianIndex`.
    ## We don't need to operate on index `i` and `j` here, so use `identity`.
    index_map = (
        identity,
        identity
    )
    ## defines:
    ##   p = A[ĩ] := @view X[:, ĩ]
    ##   q = B[j̃] := @view Y[:, j̃]
    ## Here the input A is axes(X, 2) and is not used.
    getindex_op = (
        (ax, i) -> view(X, :, i),
        (ax, j) -> view(Y, :, j)
    )
    ## defines:
    ##   out[i, j] := d(p, q)
    d = Euclidean()
    PairwiseDistance(index_map, getindex_op, d, (axes(X, 2), axes(Y, 2)))
end

#-
dist == lazy_dist

# The actual computation is not done until we retrieve the result:
#
# ```julia
# julia> @btime getindex($dist, 2, 3);
#  1.706 ns (0 allocations: 0 bytes)
#
# julia> @btime getindex($lazy_dist, 2, 3)
#  27.086 ns (0 allocations: 0 bytes)
# ```
#
# To properly benchmark the performance, let's do some simple statistics, e.g, `sum`, over the
# result.
#
# ```julia
# X = rand(1:10, 100, 200);
# Y = rand(1:10, 100, 300);
#
# function naive_pairwise_sum(d, X, Y)
#     return sum(pairwise(d, X, Y))
# end
#
# function lazy_pairwise_sum(d, X, Y)
#     getindex_op = (
#         (ax, i) -> view(X, :, i),
#         (ax, j) -> view(Y, :, j)
#     )
#     dist = PairwiseDistance(identity, getindex_op, d, map(A->axes(A, 2), (X, Y)))
#     sum(dist)
# end
#
# naive_pairwise_sum(Euclidean(), X, Y) ≈ lazy_pairwise_sum(Euclidean(), X, Y) # true
#
# @btime naive_pairwise_sum(Euclidean(), $X, $Y); # 4.513 ms (5 allocations: 473.23 KiB)
# @btime lazy_pairwise_sum(Euclidean(), $X, $Y); # 2.265 ms (13 allocations: 352 bytes)
# ```
#
# Great that we have successfully removed the memory allocation overhead and it does improve the
# performance. Of course, if you need to reuse the data repeatedly, you still need to `collect` it
# so that you don't suffer from the repeat calculation.

# ## columnwise distance
#
# Strictly speaking, `Distances.columnwise` distance is just a simple map function. Again, in
# contrast, `PairwiseDistance` does not assume the data layout, and you have to specify it 

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

# Let's also benchmark the results:
#
# ```julia
# X = rand(1:10, 100, 200);
# Y = rand(1:10, 100, 200);
#
# function naive_colwise_sum(d, X, Y)
#     return sum(colwise(d, X, Y))
# end
#
# function lazy_colwise_sum(d, X, Y)
#     getindex_op = (
#         (ax, i) -> view(X, :, i),
#         (ax, j) -> view(Y, :, j)
#     )
#     dist = PairwiseDistance(identity, getindex_op, d, map(A->axes(A, 2), (X, Y)))
#     sum(diag(dist))
# end
#
# naive_colwise_sum(Euclidean(), X, Y) ≈ lazy_colwise_sum(Euclidean(), X, Y) # true
#
# @btime naive_colwise_sum(Euclidean(), $X, $Y); # 6.002 μs (1 allocation: 1.77 KiB)
# @btime lazy_colwise_sum(Euclidean(), $X, $Y); # 8.166 μs (14 allocations: 2.11 KiB)
# ```
#
# As you can see, this time our lazy version becomes slower than the colwise version. This is
# because the `diag` operation itself gives some overhead. `PairwiseDistance` itself also
# contributes some of the overhead.
