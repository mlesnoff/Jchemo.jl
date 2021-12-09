struct Blockscal
    X::Matrix{Float64}
    blocks::DataFrame
    scal::Vector{Float64}
    list_bl
    weights::Vector{Float64}
end

"""
    blockscal(X, weights = ones(size(X, 1)); list_bl, scal = nothing)
Autoscale blocks of a matrix.
* `X` : X-data.
* `weights` : Weights of the observations.
* `list_bl` : A vector whose each component are the colum numbers
    in `X` defining a block. The length of `list_bl` is the number
    of blocks.
* `scal` : If `nothing`, each block is autoscaled (i.e. divided) by the 
    the square root of the sum of the variances of each column of the block.
    Else, `scal` must be a vector of the scaling factors
    (i.e. values dividing the blocks).

The `weights` are internally normalized to sum to 1.

Function `blockscal` can be used for instance to fit multi-block PLS
models.

## Examples
```julia
n = 5 ; p = 10 ; m = 2
X = rand(m, p) 

list_bl = [3:4, 1, [6; 8:10]]
scal = nothing
#scal = [3.1 2 .7]
res = blockscal(X; list_bl = list_bl, scal = scal) ;
res.X
res.blocks
res.scal
colvars(res.X)

Jchemo.transform(res, X) # Autoscaling of a new `X' based on `res`
```
"""
function blockscal(X, weights = ones(size(X, 1)); list_bl, scal = nothing)
    X = ensure_mat(X)
    n = size(X, 1)
    weights = mweights(weights)
    nbl = length(list_bl)
    numcol = reduce(vcat, list_bl)
    ncol = length(numcol)
    znumcol = collect(1:ncol)
    if isnothing(scal)
        zscal = similar(X, nbl)
    else
        zscal = copy(scal)
    end
    X_scal = similar(X, n, ncol)
    bl = Int64[]
    k = 1
    for i = 1:nbl
        len = length(list_bl[i])
        vX = vcol(X, list_bl[i])
        if isnothing(scal)
            zscal[i] = sqrt(sum(colvars(vX, weights)))
        end
        X_scal[:, k:(k + len - 1)] .= vX / zscal[i]
        append!(bl, fill(i, len))
        k = k + len
    end
    blocks = DataFrame([numcol znumcol bl], [:old, :new, :bl])
    Blockscal(X_scal, blocks, zscal, list_bl, weights)
end 

function transform(object::Blockscal, X)
    blockscal(X, object.weights; list_bl = object.list_bl, scal = object.scal).X
end


