"""
    PairwiseDistance(f, A, B, [cache_strategy=NullCache()])

Lazily calculate the pairwise result of two arrays `A`, `B` with function `f`.

The output `dist::PairwiseDistance` is a read-only array type, where `dist[i, j]` is defined as
`f(A[i], B[j])`.

Each item `dist[i, j]` is computed lazily during `getindex`, which could be expensive for
non-trivial `f`. `cache_strategy` provides a way to automatically store some of the result as cache.
The default strategy is `NullCache()`, which means no cache is enabled.

!!! warning
    How data is cached is internal implementation details and you generally should not talk
    to cache directly.

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
        @views A[p...], B[q...]
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
        @views A[p...], B[q...]
    end
end

Base.axes(dist::PairwiseDistance) = axes(dist.cache)
Base.size(dist::PairwiseDistance) = map(length, axes(dist))

Base.@propagate_inbounds function Base.getindex(dist::PairwiseDistance{T, N}, I::Vararg{Int, N})::T where {T, N}
    @boundscheck checkbounds(dist, I...)

    # TODO: Currently, each getindex call has about 16ns overhead, which is quite large and even
    # slows down the performance.
    if is_cached(dist.cache, I...)
        rst = @inbounds getindex(dist.cache, I...)
        if ismissing(rst)
            rst = convert(T, _evaluate_getindex(dist, I...))
            @inbounds dist.cache[I...] = rst
        end
        return convert(T, rst)
    else
        return _evaluate_getindex(dist, I...)
    end
end

Base.@propagate_inbounds function Base.getindex(
        dist::PairwiseDistance{T, N, NA, M, F, CA},
        I::Vararg{Int, N})::T where {T, N, NA, M, F, CA<:NullCacheArray}
    _evaluate_getindex(dist, I...)
end

@inline function _evaluate_getindex(dist::PairwiseDistance{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    p_val, q_val = dist.map_op(I[1:NA], I[NA+1:end])
    convert(T, dist.eval_op(p_val, q_val))
end

# To remove unnecessary information and get a clean view
function Base.showarg(io::IO, ::PairwiseDistance{T, N}, toplevel) where {T, N}
    print(io, "PairwiseDistance{", T, ", ", N, "}")
end
