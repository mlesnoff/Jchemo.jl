"""
    wilks(A::AbstractMatrix{Q}; digits = 5) where Q <: Float
Compute statistics for multivariate tests.
* `A` : Matrix involved in the test.
Keyword arguments:
* `digits` : Nb. digits for the outputs.

Return
* Wilks’ lambda
* Pillai’s trace
* Hotelling-Lawley trace
* Roy’s maximum root

## References

https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm#statug_introreg001918

## Examples 
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/reaction_bertinetto.jld2")
@load db dat
@names dat
datf = dat.datf
n = nro(datf)
tab(datf; group = [:temp, :catal])  # balanced design
##
Y = Matrix(datf[:, [:y1, :y2]])
fact = datf.temp
tab(fact)

B = matB(Y, fact, pweight(ones(n))).B
W = matW(Y, fact, pweight(ones(n))).W
@show wilks(B * inv(W))
manova(Y, @formula(0 ~ temp), datf)
```
"""
function wilks(A::AbstractMatrix{Q}; digits = 5) where Q <: Float
    v = eigen(A; sortby = x -> -abs(x)).values
    wilks = cumprod(1 ./ (1 .+ v))[end]
    pillai = sum(v ./ (1 .+ v))
    hotelling = sum(v)
    roy = v[1]
    res = round.((wilks, pillai, hotelling, roy); digits)
    (; zip([:wilks, :pillai, :hotelling, :roy], res)...)
end
