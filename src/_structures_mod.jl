Base.@kwdef mutable struct Model{T <: Function, K <: Base.Pairs}
    fun::T   
    fm
    kwargs::K
    typ::Symbol    
end

model = function(fun::Function, typ::Symbol = :x; kwargs...)
    @assert in([:x, :xy])(typ) "Wrong value for argument 'typ'."
    Model(fun, nothing, kwargs, typ)
end


