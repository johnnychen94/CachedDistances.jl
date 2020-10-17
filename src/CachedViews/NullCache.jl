struct NullCache <:AbstractCacheStrategy end
struct NullCacheArray{T, N, IA, IB} <: CachedView{T, N}
    axesA::IA
    axesB::IB
end
make_cache(T, S::NullCache, axesA, axesB, args...) = NullCacheArray{T, length(axesA)+length(axesB), typeof(axesA), typeof(axesB)}(axesA, axesB)
Base.axes(A::NullCacheArray) = (A.axesA..., A.axesB...)
Base.size(A::NullCacheArray) = map(length, axes(A))
