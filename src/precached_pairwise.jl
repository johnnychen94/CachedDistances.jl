# A even faster PairwiseDistance by
#   - eliminating unnecessary if branches via pre-caching.
#   - using faster calculating strategy for block evaluation

struct CachedPairwiseDistance{T, N, NA, PT<:PairwiseDistance, CT} <: AbstractArray{T, N}
    dist::PT
    cache::CT
    function CachedPairwiseDistance(dist::PairwiseDistance{T, N, NA}, cache=nothing) where {T,N,NA}
        new{T, N, NA, typeof(dist), typeof(cache)}(dist, cache)
    end
end

Base.size(d::CachedPairwiseDistance) = size(d.dist)
Base.axes(d::CachedPairwiseDistance) = axes(d.dist)

function get_patch(dist::CachedPairwiseDistance{T,N,NA}, p::CartesianIndex{NA}, qs::AbstractArray{<:CartesianIndex{NA}}) where {T,N,NA}
    os = qs .- p
    I_first = CartesianIndex(p.I..., first(os).I...)
    I_last = CartesianIndex(p.I..., last(os).I...)
    
    R = CartesianIndices(dist.cache)
    if _in(I_first, R) && _in(I_last, R)
        return @inbounds dist.cache[p, os]
    else
        return getindex(dist, p, qs)
    end
end

Base.@propagate_inbounds function Base.getindex(dist::CachedPairwiseDistance{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    p, q = Base.IteratorsMD.split(I, Val(NA))
    _getindex(cache_type(dist.dist), dist, CartesianIndex(p), CartesianIndex(q))
end

@inline function _getindex(
    ::Type{CA},
    dist::CachedPairwiseDistance{T, N, NA},
    p::CartesianIndex{NA}, q::CartesianIndex{NA}) where {T, N, NA, CA<:NullCacheArray}
    _evaluate_getindex(dist.dist, p, q)
end

@inline function _getindex(
    ::Type{CA},
    dist::CachedPairwiseDistance{T, N, NA},
    p::CartesianIndex{NA}, q::CartesianIndex{NA})::T where {T, N, NA, CA<:LocalWindowCacheArray}
    o = q - p
    I = CartesianIndex(p.I..., o.I...)

    # Cache might be smaller than the actual array size
    if _in(I, CartesianIndices(dist.cache))
        return @inbounds dist.cache[I]
    else
        return _evaluate_getindex(dist.dist, p, q)
    end
end

function precalculate(d::PairwiseDistance{T, N, NA, CA}) where {T, N, NA, CA<:LocalWindowCacheArray}
    @assert N == 2NA "more than 2 arrays is not supported."
    R = CartesianIndices(d)

    # initialize cache using the center point
    I = R[CartesianIndex(size(R).÷2)]
    p, q = CartesianIndex.(Base.IteratorsMD.split(I.I, Val(NA)))
    buffer = collect(d.getindex_op[2](d.As[2], _to_indices(d.index_map[2](q))))

    A, B = d.As
    cache = similar(d.cache.cache, T)
    R_A = CartesianIndices(A)
    R_B = CartesianIndices(B)
    R_cache = CartesianIndices(cache)

    index_map_A, index_map_B = d.index_map
    getindex_op_A, getindex_op_B = d.getindex_op
    f = d.f
    for l in axes(d, 4)
        for k in axes(d, 3)
            q = CartesianIndex(k, l)
            patch_q_inds = _to_indices(index_map_B(q))
            first(patch_q_inds) in R_B && last(patch_q_inds) in R_B || continue

            buffer = getindex_op_B(B, patch_q_inds)
            for j in axes(d, 2)
                for i in axes(d, 1)
                    p = CartesianIndex(i, j)
                    patch_p_inds = _to_indices(index_map_A(p))
                    first(patch_p_inds) in R_A && last(patch_p_inds) in R_A || continue

                    o = q - p
                    i_buffer = CartesianIndex(p.I..., o.I...)
                    i_buffer in R_cache || continue

                    patch_p = getindex_op_A(A, patch_p_inds)
                    value = f(buffer, patch_p)
                    @inbounds cache[i_buffer] = value
                end
            end
        end
    end

    return CachedPairwiseDistance(d, cache)
end
