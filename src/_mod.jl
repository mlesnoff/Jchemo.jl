Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fm
    kwargs::K
end

"""
    mod_(algo::Function; kwargs...)
Build a model.
* `algo` : The algorthm (function) defining the model.
* `kwargs...`: Keyword arguments of `algo`.

## Examples
```julia
X = rand(5, 10)
y = rand(5)

model = mod_(dtpol)  # use the default arguments of 'dtpol'
#model = dtpol(X; degree = 2)
pnames(model)
fit!(model, X)
Xp = transf(model, X)

model = mod_(plskern; nlv = 3) 
fit!(model, X, y)
pred = predict(model, X).pred
```
"""
function mod_(algo::Function; kwargs...)
    JchemoModel(algo, nothing, kwargs)
end
