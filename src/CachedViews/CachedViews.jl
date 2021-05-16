module CachedViews

using OffsetArrays

abstract type CachedView{T, N} <: AbstractArray{T, N} end
abstract type AbstractCacheStrategy end

export NullCache, LocalWindowCache

include("NullCache.jl")
include("LocalWindowCache.jl")

end # module
