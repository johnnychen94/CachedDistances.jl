module CachedViews

using OffsetArrays
import Base: size

# The cache view protocol
#
# * `make_cache` creates a CachedView
# * `in_cache` to check whether this is cached in `A`
# * `getindex(A, i)`: return `missing` if `A[i]` is not cached
# * `setindex!(A, v, i)`: to cache the data `A[i] = v`
#
# This protocol only specifies how cache data is retrived, whether the cache data is up-to-date is
# not in the scope of it and it's all depends on package user to define the behavior.
#
# WIP: This protocol is only used to compute pairwise distances, there might be changes to make it
# applicable to other tasks.

abstract type CachedView{T, N} <: AbstractArray{T, N} end
abstract type AbstractCacheStrategy end

"""
    is_cached(A::CachedView, i...)::Bool

Check whether the value at index `i` is cached in `A`.

This check does not guarantee that cached value is up-to-date.
"""
is_cached(A::CachedView, i::Int) = is_cached(A, Base._to_subscript_indices(A, i)...)

export NullCache, LocalWindowCache

include("NullCache.jl")
include("LocalWindowCache.jl")

end # module
