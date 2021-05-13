"""
    PairwiseDistance([index_map], [getindex_op], f, A, B, [cache_strategy])

Lazily calculate the pairwise result of two array-like objects `A`, `B` with function `f`.

The output `dist` is a read-only array with `dist[i, j]` defined as `f(p, q)`, where
`ĩ, j̃ = index_map(i), index_map(j)`, and `p, q = getindex_op(A, ĩ), getindex_op(B, j̃)`.

# Optional Arguments

- `index_map=identity`:
  `index_map` is a function-like object that maps given index into a new index or indices.
- `getindex_op=@view getindex`:
  `getindex_op` is a funtion-like object that defines how `p` and `q` are get. By default it is
  a viewed version of getindex (without data copy).
  If `getindex_op` is provided, then `index_map` should be provided, too.
- `cache_strategy=NullCache()`:
  Specify the cache strategy.

# Cache Strategies

For non-trivial `f`, it could be expensive if we calculate it everytime when `dist[i, j]` is required.
`cache_strategy` provides a way to automatically store some of the result in cache so that
`f(A[i], B[j])` only gets called once in specific cases.

Current available cache strategies are:

- [`NullCache()`](@ref): no cache behavior. Every time when you indexing `dist[i, j]` you're calculating
  `f(A[i], B[j])`.
- [`LocalWindowCache(window_size)`](@ref): cache a local window of `i` so that `f(A[i], B[j])` only
  gets called once if `j ∈ i-r:i+r`, and gets called multiple times otherwise.

!!! warning
    How data is cached is internal implementation details. Generally, you should not directly talk
    to cache.

# Examples

```jldoctest; setup=:(using Random; Random.seed!(0))
julia> using LazyDistances, Distances

julia> A, B = 1:6, 1:4

julia> dist = PairwiseDistance(Euclidean(), A, B)
6×4 PairwiseDistance{Float64, 2}:
 0.0  1.0  2.0  3.0
 1.0  0.0  1.0  2.0
 2.0  1.0  0.0  1.0
 3.0  2.0  1.0  0.0
 4.0  3.0  2.0  1.0
 5.0  4.0  3.0  2.0

# For each `i`, `dist[i, i-2:i+2]` will be cached, which means `f(A[i], B[j])` for `j ∈ i-2:i+2`
# will only be calculated once.
julia> dist = PairwiseDistance(Euclidean(), A, B, LocalWindowCache((5, )))
6×4 PairwiseDistance{Float64, 2}:
 0.0  1.0  2.0  3.0
 1.0  0.0  1.0  2.0
 2.0  1.0  0.0  1.0
 3.0  2.0  1.0  0.0
 4.0  3.0  2.0  1.0
 5.0  4.0  3.0  2.0
```
"""
struct PairwiseDistance{T, N, NA, M, G, F, AT, BT, CA<:CachedView} <: AbstractArray{T, N}
    index_map::M
    getindex_op::G
    f::F
    A::AT
    B::BT
    cache::CA
    function PairwiseDistance{T}(
            index_map::M, getindex_op::G, f::F,
            A::AbstractArray{<:Any, NA}, B::AbstractArray{<:Any, NA},
            cache_strategy::AbstractCacheStrategy=NullCache()) where {T,NA,M,G,F}

        cache = make_cache(T, cache_strategy, axes(A), axes(B))
        new{T, 2NA, NA,M, G, F, typeof(A), typeof(B), typeof(cache)}(
            index_map, getindex_op, f, A, B, cache
        )
    end
end

function PairwiseDistance(
        index_map, getindex_op, f,
        A::AbstractArray, B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache())
    try
        T = result_type(f, eltype(A), eltype(B))
        return PairwiseDistance{T}(index_map, getindex_op, f, A, B, cache_strategy)
    catch e
        @error "Unable to infer the result type, please specify it explicitly using `PairwiseDistance{T}`
                method."
        rethrow(e)
    end
end
function PairwiseDistance(
        index_map, f,
        A::AbstractArray, B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache())
    # If only two functions are provided, assume it's `index_map` and `f`.
    PairwiseDistance(index_map, _getindex_view, f, A, B, cache_strategy)
end
function PairwiseDistance(
        f,
        A::AbstractArray, B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache())
    PairwiseDistance(identity, _getindex_view, f, A, B, cache_strategy)
end

function PairwiseDistance{T}(
        index_map, f,
        A::AbstractArray,
        B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache()) where T
    PairwiseDistance{T}(index_map, _getindex_view, f, A, B, cache_strategy)
end
function PairwiseDistance{T}(
        f,
        A::AbstractArray,
        B::AbstractArray,
        cache_strategy::AbstractCacheStrategy=NullCache()) where T
    PairwiseDistance{T}(identity, _getindex_view, f, A, B, cache_strategy)
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

Base.@propagate_inbounds function _evaluate_getindex(d::PairwiseDistance{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    i, j = I[1:NA], I[NA+1:end]
    p = d.getindex_op(d.A, _to_indices(d.index_map(i)))
    q = d.getindex_op(d.B, _to_indices(d.index_map(j)))
    convert(T, d.f(p, q))
end

# To remove unnecessary information and get a clean view
function Base.showarg(io::IO, ::PairwiseDistance{T, N}, toplevel) where {T, N}
    print(io, "PairwiseDistance{", T, ", ", N, "}")
end

# This unifies the way to index `A[ĩ]` (so that we never need to explicitly call `A[ĩ...]`)
@inline _to_indices(inds::Tuple) = CartesianIndices(inds)
@inline _to_indices(inds::Dims) = CartesianIndex(inds)
@inline _to_indices(i::Union{Integer, CartesianIndex}) = i
@inline _to_indices(R::CartesianIndices) = R

Base.@propagate_inbounds _getindex_view(A::AbstractArray, R::CartesianIndices) = @view A[R]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, inds::Dims) = @view A[inds...]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, inds::Int...) = @view A[inds...]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, i::Union{Integer, CartesianIndex}) = A[i]
