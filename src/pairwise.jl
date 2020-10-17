"""
    PairwiseDistance([map_op,] f, A, B, [cache_strategy=NullCache()])

Lazily calculate the pairwise result of two arrays `A`, `B` with function `f`.

The output `dist::PairwiseDistance` is a read-only array type, where `dist[p, q]` is defined as
`f(A[p], B[q])`. Each item `dist[p, q]` is computed lazily during `getindex`, which could be
expensive for non-trivial `f`. `cache_strategy` provides a way to automatically store some of the
result as cache.

`map_op(p::CartesianIndex{N}, q::CartesianIndex{N}) where N` is used to map `p` in `A` and `q` in
`B` to other data, e.g., column, row, block. The default version is `getindex`, i.e., `map_op(p, q)
= A[p], B[q]`. The output of `map_op` is the input of `f` and thus you have to make sure these two
function works seamlessly.

Two cache types are available:

- `NullCache()` (default): no cache is used, every `getindex` does the actual computation.
- `LocalWindowCache(window_size)`: use a cache array of size `(size(A)..., window_size...)`. For
   each point `p`, it caches the results `f(A[p], B[q])` where `q ∈ p-r:p+r`, and `r` is the cache
   window radius. This cache strategy has strong assumption on the spatial locality. This cache
   strategy currently has an `getindex` overhead about 4-6ns and thus should not be applied to
   trivial computations.

# Examples

```jldoctest; setup=:(using Random; Random.seed!(0))
julia> using LazyDistances, Distances

julia> A, B = rand(1:5, 6), rand(1:5, 4);

julia> dist = PairwiseDistance(Euclidean(), A, B)
6×4 PairwiseDistance{Float64, 2}:
 1.0  1.0  2.0  2.0
 4.0  4.0  1.0  1.0
 2.0  2.0  1.0  1.0
 4.0  4.0  1.0  1.0
 0.0  0.0  3.0  3.0
 2.0  2.0  1.0  1.0

julia> dist = PairwiseDistance(Euclidean(), A, B, LocalWindowCache((3, )))
6×4 PairwiseDistance{Float64, 2}:
 1.0  1.0  2.0  2.0
 4.0  4.0  1.0  1.0
 2.0  2.0  1.0  1.0
 4.0  4.0  1.0  1.0
 0.0  0.0  3.0  3.0
 2.0  2.0  1.0  1.0
```
"""
struct PairwiseDistance{T, N, NA, M, F, CA<:CachedView} <: AbstractArray{T, N}
    map_op::M
    eval_op::F
    cache::CA
end

function PairwiseDistance(map_op, eval_op,
        A::AbstractArray{T1, N},
        B::AbstractArray{T2, N},
        cache_strategy::AbstractCacheStrategy=NullCache()) where {T1, T2, N}

    T = result_type(eval_op, T1, T2)
    cache = make_cache(T, cache_strategy, axes(A), axes(B))
    PairwiseDistance{T, 2N, N, typeof(map_op), typeof(eval_op), typeof(cache)}(map_op, eval_op, cache)
end

function PairwiseDistance(eval_op, A::AbstractArray, B::AbstractArray, cache_strategy::AbstractCacheStrategy=NullCache())
    PairwiseDistance(eval_op, A, B, cache_strategy) do p, q
        A[p], B[q]
    end
end

function PairwiseDistance{T}(map_op, eval_op,
        A::AbstractArray{T1, N},
        B::AbstractArray{T2, N},
        cache_strategy::AbstractCacheStrategy=NullCache()) where {T, T1, T2, N}
    cache = make_cache(T, cache_strategy, axes(A), axes(B))
    PairwiseDistance{T, 2N, N, typeof(map_op), typeof(eval_op), typeof(cache)}(map_op, eval_op, cache)
end

function PairwiseDistance{T}(eval_op,
        A::AbstractArray,
        B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache()) where T
    PairwiseDistance{T}(eval_op, A, B, cache_strategy) do p, q
        A[p], B[q]
    end
end

Base.axes(dist::PairwiseDistance) = axes(dist.cache)
Base.size(dist::PairwiseDistance) = map(length, axes(dist))

Base.@propagate_inbounds _evaluate_getindex(dist, p, q) = dist.eval_op(dist.map_op(p, q)...)

Base.@propagate_inbounds function Base.getindex(dist::PairwiseDistance{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    p, q = Base.IteratorsMD.split(I, Val(NA))
    _getindex(dist, CartesianIndex(p), CartesianIndex(q))
end

Base.@propagate_inbounds function _getindex(
        dist::PairwiseDistance{T, N, NA, M, F, CA},
        p::CartesianIndex{NA}, q::CartesianIndex{NA}) where {T, N, NA, M, F, CA<:NullCacheArray}
    _evaluate_getindex(dist, p, q)
end

Base.@propagate_inbounds function _getindex(
        dist::PairwiseDistance{T, N, NA, M, F, CA},
        p::CartesianIndex{NA}, q::CartesianIndex{NA})::T where {T, N, NA, M, F, CA<:LocalWindowCacheArray}
    o = q - p
    I = CartesianIndex(p.I..., o.I...)

    # Cache might be smaller than the actual array size
    if _in(I, dist.cache.R)
        rst = @inbounds dist.cache.cache[I]
        if ismissing(rst)
            rst = _evaluate_getindex(dist, p, q)
            @inbounds dist.cache.cache[I] = rst
        end
        return rst
    else
        return _evaluate_getindex(dist, p, q)
    end
end

# To remove unnecessary information and get a clean view; cache should be transparent to user
function Base.showarg(io::IO, ::PairwiseDistance{T, N}, toplevel) where {T, N}
    print(io, "PairwiseDistance{", T, ", ", N, "}")
end
