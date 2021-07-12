"""
    xfit(object::Union{Pca, Plsr}, X; nlv = nothing)
Matrix fitting from a PCA or PLS model
* `object` : The fitted model.
* `X` : X-data to be approximatred from the model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum nb. of components.

Function `xfit` computes an approximate of matrix `X` (X_fit) from a PCA or PLS fitted on `X`.
""" 
function xfit(object::Union{Pca, Plsr}, X; nlv = nothing)
    xfit!(object, copy(X); nlv = nlv)
end

function xfit!(object::Union{Pca, Plsr}, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    if(nlv == 0)
        m = size(X, 1)
        @inbounds for i = 1:m
            X[i, :] .= object.xmeans
        end
    else
        isa(object, Jchemo.Pca) ? R = @view(object.P[:, 1:nlv]) : R = @view(object.R[:, 1:nlv])
        X .= transform(object, X; nlv = nlv) * R'
        center!(X, -object.xmeans)
    end
    X
end

"""
    xresid(object::Union{Pca, Plsr}, X; nlv = nothing)
Residual matrix after fitting by a PCA or PLS model
* `object` : The fitted model.
* `X` : X-data for which the residuals have to be computed.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum nb. of components.

Function `xresid` computes the residual matrix E = X - X_fit.
""" 
function xresid(object::Union{Pca, Plsr}, X; nlv = nothing)
    xresid!(object, copy(X); nlv = nlv)
end

function xresid!(object::Union{Pca, Plsr}, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    if(nlv == 0)
        center!(X, object.xmeans)
    else
        X .= X .- xfit(object, X; nlv = nlv)
    end
    X
end



