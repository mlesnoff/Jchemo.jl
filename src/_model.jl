Base.@kwdef mutable struct Model{T <: Function, K <: Base.Pairs}
    fun::T   
    fm
    kwargs::K
end

"""
    model(fun::Function; kwargs...)
Build a model.
* `fun` : The function defining the the model.
* `kwargs...`: Keyword arguments of `fun`.

## Examples
```julia
X = rand(5, 10)
y = rand(5)

mod = model(dtpol)  # use the default arguments of 'dtpol'
#mod = dtpol(X; degree = 2)
pnames(mod)
fit!(mod, X)
Xp = transf(mod, X)

mod = model(plskern; nlv = 3) 
fit!(mod, X, y)
pred = predict(mod, X).pred
```
"""
function model(fun::Function; kwargs...)
    Model(fun, nothing, kwargs)
end
