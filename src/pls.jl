struct Pls7
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Px::Matrix{Float64}
    Py::Matrix{Float64}
    Rx::Matrix{Float64}
    Ry::Matrix{Float64}    
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    TTx::Vector{Float64}
    TTy::Vector{Float64}
    delta::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

function pls(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", scal = false)
    pls!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        bscal = bscal, scal = scal)
end

function pls!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    D = Diagonal(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    bscal == "none" ? bscales = ones(2) : nothing
    if bscal == "frob"
        normx = fnorm(X, weights)
        normy = fnorm(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    # Pre-allocation
    XtY = similar(X, p, q)
    Tx = similar(X, n, nlv)
    Ty = copy(Tx)
    Wx = similar(X, p, nlv)
    Wy = similar(X, q, nlv)
    Px = copy(Wx)
    Py = copy(Wy)
    TTx = similar(X, nlv)
    TTy = copy(TTx)
    tx   = similar(X, n)
    ty = copy(tx)
    dtx  = copy(tx)
    dty = copy(tx)   
    wx  = similar(X, p)
    wy  = similar(X, q)
    px   = copy(wx)
    py   = copy(wy)
    delta = copy(TTx)
    # End
    @inbounds for a = 1:nlv
        XtY .= X' * (D * Y)
        U, d, V = svd!(XtY) 
        delta[a] = d[1]
        # X
        wx .= U[:, 1]
        mul!(tx, X, wx)
        dtx .= weights .* tx
        ttx = dot(tx, dtx)
        mul!(px, X', dtx)
        px ./= ttx
        # Y
        wy .= V[:, 1]
        # Same as:                        
        # mul!(wy, Y', dtx)
        # wy ./= norm(wy)
        # End
        mul!(ty, Y, wy)
        dty .= weights .* ty
        tty = dot(ty, dty)
        mul!(py, Y', dty)
        py ./= tty
        # deflation
        X .-= tx * px'
        Y .-= ty * py'
        # If regression mode
        #ctild = Y' * dty / tty
        #Y .-= tx * ctild'
        # End
        #u = svd([tx ty]).U[:, 1]
        #u = tx + ty ; u ./= norm(u)
        #X .-= u * (u' * X)
        #Y .-= u * (u' * Y)
        # End
        Px[:, a] .= px
        Py[:, a] .= py
        Tx[:, a] .= tx
        Ty[:, a] .= ty
        Wx[:, a] .= wx
        Wy[:, a] .= wy
        TTx[a] = ttx
        TTy[a] = tty
     end
     Rx = Wx * inv(Px' * Wx)
     Ry = Wy * inv(Py' * Wy)
     Pls7(Tx, Ty, Px, Py, Rx, Ry, Wx, Wy, TTx, TTy, delta, 
         bscales, xmeans, xscales, ymeans, yscales, weights)
end

function transform(object::Pls7, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Rx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Ry, 1:nlv)
    (Tx = Tx, Ty)
end

function Base.summary(object::Pls7, X::Union{Vector, Matrix, DataFrame},
        Y::Union{Vector, Matrix, DataFrame})
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales)
    Y = cscale(Y, object.ymeans, object.yscales)
    ttx = object.TTx 
    tty = object.TTy 
    ## X
    sstot = fnorm(X, object.weights)^2
    tt_adj = vec(sum(object.Px.^2, dims = 1)) .* ttx
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## y
    sstot = fnorm(Y, object.weights)^2
    tt_adj = vec(sum(object.Py.^2, dims = 1)) .* tty
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## End
    (explvarx = explvarx, explvary)
end

