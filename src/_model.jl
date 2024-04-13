Base.@kwdef mutable struct Model{T <: Function, K <: Base.Pairs}
    fun::T   
    fm
    kwargs::K
    typ::Union{Nothing, Symbol}    
end

function model(fun::Function; kwargs...)
    Model(fun, nothing, kwargs, nothing)
end

function model(fun::Function, typ::Symbol; kwargs...)
    Model(fun, nothing, kwargs, typ)
end

function modelx(fun::Function; kwargs...)
    Model(fun, nothing, kwargs, :x)
end

function modelxy(fun::Function; kwargs...)
    Model(fun, nothing, kwargs, :xy)
end

