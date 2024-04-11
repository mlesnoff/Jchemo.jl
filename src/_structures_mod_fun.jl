Base.@kwdef mutable struct Model1{FUN <: Function, FM, KWARGS <: Base.Pairs}
    fun::FUN   
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end

Base.@kwdef mutable struct Model
    fun::Function   
    fm
    kwargs    
end

model = function(fun::Function; kwargs...)
    Model(fun, nothing, kwargs)
end



