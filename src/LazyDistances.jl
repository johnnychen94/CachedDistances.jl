module LazyDistances

using Reexport
using Distances

include("CachedViews/CachedViews.jl")
@reexport using .CachedViews
using .CachedViews: CachedView, AbstractCacheStrategy, NullCacheArray, LocalWindowCacheArray, make_cache

export PairwiseDistance

# extend Distances.result_type
# https://github.com/JuliaStats/Distances.jl/pull/185
result_type(f, a::Type, b::Type) = typeof(f(oneunit(a), oneunit(b)))
result_type(f::Distances.PreMetric, a::Type, b::Type) = Distances.result_type(f, a, b) # ambiguity patch
result_type(f, A::AbstractArray, b::AbstractArray) = result_type(f, eltype(A), eltype(B))
result_type(dist::Distances.PreMetric, a, b) = Distances.result_type(dist, a, b)

include("utilities.jl")
include("pairwise.jl")
include("precached_pairwise.jl")

end #module
