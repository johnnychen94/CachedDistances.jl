# ---
# title: Row/Column-wise Distances
# author: Johnny Chen
# date: 2020-10-16
# ---

# This example shows how `LazyDistances.PairwiseDistance` can be used to get a better
# `Distances.pairwise` performance.

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
    T = result_type(Euclidean(), X, Y)
    PairwiseDistance{T}(Euclidean(), axes(X, 2), axes(Y, 2)) do i, j
        @views X[:, i...], Y[:, j...]
    end
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
#  73.290 ns (0 allocations: 0 bytes)
# ```
#
# To properly benchmark the performance, let's do some simple statistics, e.g, `sum`, over the
# result.
#
# ```julia
# julia> X = rand(1:10, 100, 200);
#
# julia> Y = rand(1:10, 100, 300);
#
# julia> @btime sum(pairwise(Euclidean(), $X, $Y))
#  4.615 ms (5 allocations: 473.23 KiB)
#
# julia> @btime let X=$X, Y=$Y
#            T = result_type(Euclidean(), X, Y)
#            PairwiseDistance{T}(Euclidean(), axes(X, 2), axes(Y, 2)) do i, j
#                @views X[:, i...], Y[:, j...]
#            end |> sum
#        end
#  4.109 ms (0 allocations: 0 bytes)
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
    T = result_type(Euclidean(), X, Y)
    eval_op(i, j) = Euclidean()(view(X, :, i...), view(Y, :, j...))
    PairwiseDistance{Float64}(eval_op, axes(X, 2), axes(Y, 2))
end

Diagonal(lazy_dist) == Diagonal(dist)

# ```julia
# julia> X = rand(1:10, 100, 200);
# 
# julia> Y = rand(1:10, 100, 200);
# 
# julia> @btime sum(colwise(Euclidean(), $X, $Y))
#   14.513 μs (1 allocation: 1.77 KiB)
#
# julia> @btime let X=$X, Y=$Y
#            T = result_type(Euclidean(), X, Y)
#            eval_op(i, j) = Euclidean()(view(X, :, i...), view(Y, :, j...))
#            PairwiseDistance{T}(eval_op, axes(X, 2), axes(Y, 2)) |> Diagonal |> sum
#        end
#   15.207 μs (1 allocation: 1.77 KiB)
# ```
