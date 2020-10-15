struct NullCache <:AbstractCacheStrategy end
struct NullCacheArray{T, N, IA, IB} <: CachedView{T, N}
    axesA::IA
    axesB::IB
end
make_cache(T, S::NullCache, axesA, axesB, args...) = NullCacheArray{T, length(axesA)+length(axesB), typeof(axesA), typeof(axesB)}(axesA, axesB)
is_cached(::NullCacheArray, ::Int...) = false
getindex(::NullCacheArray{T, N}, ::Vararg{Int, N}) where {T, N} = missing
setindex!(::NullCacheArray{T, N}, v, ::Vararg{Int, N}) where {T, N} = v
Base.axes(A::NullCacheArray) = (A.axesA..., A.axesB...)
Base.size(A::NullCacheArray) = map(length, axes(A))
