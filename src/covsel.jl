function covsel(X, Y; nlv = nothing, meth::Symbol = :cov, scal::Bool = false)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    covsel(X, Y, weights; nlv, meth, scal)
end

function covsel(X, Y, weights::Jchemo.Weight; nlv = nothing, meth::Symbol = :cov, scal::Bool = false)
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv, meth, scal)
end

function covsel!(X::Matrix, Y::Matrix, weights::Jchemo.Weight; nlv = nothing, meth::Symbol = :cov, 
        scal::Bool = false)
    par = (nlv = nlv, meth, scal) # to be replaced by recovkw in the future
    ## Specific for Da functions
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    ## End 
    n, p = size(X)
    q = nco(Y)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)  
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        #fcscale!(Y, ymeans, yscales)
        fcenter!(Y, ymeans)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    sqrtw = sqrt.(weights.w)
    fweight!(X, sqrtw)
    fweight!(Y, sqrtw)
    xsstot = Jchemo.frob2(X)
    ysstot = Jchemo.frob2(Y)
    ##
    zX = copy(X)
    zY = copy(Y)
    sel = zeros(Int, nlv)
    sel_c = similar(X, nlv)
    x = similar(X, n, 1)
    XtY = similar(X, p, q)
    c = similar(X, p)
    B = similar(X, nlv, q)
    xss = copy(sel_c)
    yss = copy(sel_c)
    for i = 1:nlv
        XtY .= X' * Y    # (p, q) 
        if q == 1
            c .= XtY.^2
        else
            c .= rowsum(XtY.^2)
        end
        if par.meth == :cor
            xscales .= colnorm(X).^2
            if i > 1
                xscales[sel[1:(i - 1)]] .= 1  # remove divisions by zeros
            end
            c ./= xscales
        end
        sel[i] = argmax(c)
        sel_c[i] = c[sel[i]]
        x .= vcol(X, sel[i])
        dotx = dot(x, x) 
        ## Projecion matrix on x (n, n) = x * inv(x' * x) * x' = x * x' / dot(x, x)
        X .-= x * x' * X / dotx   
        Y .-= x * x' * Y / dotx 
        xss[i] = Jchemo.frob2(X)
        yss[i] = Jchemo.frob2(Y)
    end
    (sel = sel, sel_c, xss, yss, xsstot, ysstot, xmeans, xscales, ymeans, yscales, weights)
end

function transf_covsel(object, X)
    X[:, object.sel]
end

function summary_covsel(object)
    nlv = length(object.sel)
    xsstot = object.xsstot
    ysstot = object.ysstot
    cumpvarx = 1 .- object.xss / xsstot
    cumpvary = 1 .- object.yss / ysstot
    pvarx = cumpvarx - [0; cumpvarx[1:(nlv - 1)]]
    pvary = cumpvary - [0; cumpvary[1:(nlv - 1)]]
    explvarx = DataFrame(nlv = 1:nlv, sel = object.sel, pvar = pvarx, cumpvar = cumpvarx)
    explvary = DataFrame(nlv = 1:nlv, sel = object.sel, pvar = pvary, cumpvar = cumpvary)
    (explvarx = explvarx, explvary)
end


