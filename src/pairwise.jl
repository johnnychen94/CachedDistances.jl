"""
    PairwiseDistance([index_map], [getindex_op], f, As, [cache_strategy])

Lazily calculate the pairwise result of array-like objects `As` with function `f`.

The output `dist` is a read-only array with `dist[i, j, ...]` defined as `f(p, q, ...)`, where
`ĩ = index_map(i)`, and `p = getindex_op(As[1], ĩ)`. More formally speaking,

```julia
dist[inds...] = f(map(getindex_op, index_map, inds, As) do proj, σ, i, A
    proj(A, σ(i))
end)
```

# Required Arguments

- `f`: A function or function-like object.
- `As`: A tuple of arrays.

# Optional Arguments

- `index_map`:
  `index_map` is a function or function-like object that maps given index into a new index or indices.
  By default it is `identity`.
- `getindex_op`:
  `getindex_op` is a function or funtion-like object that defines how `p` and `q` are get.
  By default it is a viewed version of getindex (without data copy). 
  If `getindex_op` is provided, then `index_map` should be provided, too.
- `cache_strategy=NullCache()`:
  Specify the cache strategy.

!!! tips
    `index_map` and `getindex_op` can optionally be a tuple of length `length(As)`. In this case,
    `index_map[i]` and `getindex_op[i]` will applied to `As[i]`.

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
struct PairwiseDistance{T, N, NA, CA<:CachedView, M, G, F, TT} <: AbstractArray{T, N}
    index_map::M
    getindex_op::G
    f::F
    As::TT
    cache::CA
    function PairwiseDistance{T}(
            index_map::Tuple, getindex_op::Tuple, f::F,
            As::NTuple{K, <:AbstractArray}, # for simplicity, assume As is homogeneous
            cache_strategy::AbstractCacheStrategy=NullCache()) where {T,F,K}

        # TODO: support more array inputs... (Maybe...)
        length(As) == 2 || throw(ArgumentError("Currently only two arrays are supported."))

        NA = ndims(first(As))
        all(isequal(NA), ndims.(As)) || throw(ArgumentError("`As` should be of the same dimension."))

        cache = make_cache(T, cache_strategy, axes.(As))
        new{T, K*NA, NA, typeof(cache), typeof(index_map), typeof(getindex_op), F, typeof(As)}(
            index_map, getindex_op, f, As, cache
        )
    end
end

function PairwiseDistance(
        index_map, getindex_op, f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache())
    local T
    try
        T = result_type(f, eltype.(As)...)
    catch e
        @error "Unable to infer the result type, please specify it explicitly using
                the `PairwiseDistance{T}` althernative."
        rethrow(e)
    end
    return PairwiseDistance{T}(index_map, getindex_op, f, As, cache_strategy)
end

function PairwiseDistance(
        index_map, f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache())
    # If only two functions are provided, assume it's `index_map` and `f`.
    PairwiseDistance(index_map, _getindex_view, f, As, cache_strategy)
end

function PairwiseDistance(
        f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache())
    PairwiseDistance(identity, _getindex_view, f, As, cache_strategy)
end

function PairwiseDistance{T}(
        index_map, f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache()) where T
    PairwiseDistance{T}(index_map, _getindex_view, f, As, cache_strategy)
end

function PairwiseDistance{T}(
        f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache()) where T
    PairwiseDistance{T}(identity, _getindex_view, f, As, cache_strategy)
end

function PairwiseDistance{T}(
        index_map::M, getindex_op::G, f,
        As::NTuple,
        cache_strategy::AbstractCacheStrategy=NullCache()) where {T,M,G}
    index_map isa Tuple || (index_map = ntuple(_->index_map, length(As)))
    getindex_op isa Tuple || (getindex_op = ntuple(_->getindex_op, length(As)))
    PairwiseDistance{T}(index_map, getindex_op, f, As, cache_strategy)
end

Base.axes(dist::PairwiseDistance) = axes(dist.cache)
Base.size(dist::PairwiseDistance) = map(length, axes(dist))

Base.@propagate_inbounds function Base.getindex(dist::PairwiseDistance{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    p, q = Base.IteratorsMD.split(I, Val(NA))
    _getindex(dist, CartesianIndex(p), CartesianIndex(q))
end

Base.@propagate_inbounds function _getindex(
        dist::PairwiseDistance{T, N, NA, CA},
        p::CartesianIndex{NA}, q::CartesianIndex{NA}) where {T, N, NA, CA<:NullCacheArray}
    _evaluate_getindex(dist, p, q)
end

Base.@propagate_inbounds function _getindex(
        dist::PairwiseDistance{T, N, NA, CA},
        p::CartesianIndex{NA}, q::CartesianIndex{NA})::T where {T, N, NA, CA<:LocalWindowCacheArray}
    o = q - p
    I = CartesianIndex(p.I..., o.I...)

    # Cache might be smaller than the actual array size
    if _in(I, CartesianIndices(dist.cache.cache))
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

Base.@propagate_inbounds function _evaluate_getindex(d::PairwiseDistance{T, N, NA}, inds::CartesianIndex{NA}...) where {T, N, NA}
    values = map(d.index_map, d.getindex_op, d.As, inds) do index_map, getindex_op, A, i
        getindex_op(A, _to_indices(index_map(i)))
    end
    d.f(values...)
end

# To remove unnecessary information and get a clean view
function Base.showarg(io::IO, ::PairwiseDistance{T, N}, toplevel) where {T, N}
    print(io, "PairwiseDistance{", T, ", ", N, "}")
end

# This unifies the way to index `A[ĩ]` (so that we never need to explicitly call `A[ĩ...]`)
@inline _to_indices(inds::Tuple) = CartesianIndices(inds)
@inline _to_indices(inds::Dims) = CartesianIndex(inds)
@inline _to_indices(i::Union{Integer, CartesianIndex}) = i
@inline _to_indices(R::Union{CartesianIndices, AbstractArray{<:CartesianIndex}}) = R

Base.@propagate_inbounds _getindex_view(A::AbstractArray, R::Union{CartesianIndices, AbstractArray{<:CartesianIndex}}) = @view A[R]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, inds::Dims) = @view A[inds...]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, inds::Int...) = @view A[inds...]
Base.@propagate_inbounds _getindex_view(A::AbstractArray, i::Union{Integer, CartesianIndex}) = A[i]
