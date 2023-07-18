function plsrstack(X, y, weights = ones(nro(X)); nlv,
        K = 5, rep = 10, scal::Bool = false)
    plsrstack!(copy(ensure_mat(X)), copy(ensure_mat(y)), weights; nlv = nlv, 
        K = K, rep = rep, scal = scal)
end

function plsrstack!(X::Matrix, y::Matrix, weights = ones(nro(X)); nlv, 
        K = 5, rep = 10, scal::Bool = false)
    n, p = size(X)
    weights = mweight(weights)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    segm = segmkf(n, K; rep = rep)
    ystack = []
    Xstack = []
    ## Replications and segments are concatenated vertically
    ## ==> Xstack (y predictions), ystack (y observed values)
    ## Corresponding observation weights also ==> weightsstack 
    k = 1
    for i = 1:nro(segm)
        for j = 1:nro(segm[1])
            s = segm[i][j]
            Xcal = rmrow(X, s)
            ycal = rmrow(y, s)
            wcal = rmrow(weights, s)
            Xval = vrow(X, s)
            yval = vrow(y, s)
            wval = vrow(weights, s)
            zfm = plskern(Xcal, ycal, wcal; nlv =  nlvmax, scal = scal) ;
            pred = Jchemo.predict(zfm, Xval; nlv = 0:nlvmax).pred
            pred = reduce(hcat, pred)
            if k == 1
                Xstack = copy(pred) 
                ystack = copy(yval)
                weightsstack = copy(wval)
            else
                Xstack = vcat(Xstack, pred) 
                ystack = vcat(ystack, yval)
                weightsstack = vcat(weightsstack, wval)
            end
            k = k + 1
        end
    end
    ## Computation of the stacking weights (w)
    ## Linear model without intercept
    weightsstack = mweight(weightsstack)
    Xstack = vcol(Xstack, nlv .+ 1)
    XtD = Xstack' * Diagonal(weightsstack)
    w = vec(cholesky!(Hermitian(XtD * Xstack)) \ (XtD * ystack))
    ## Model that will be "averaged" (w)
    fm = plskern!(X, y, weights; nlv = nlvmax, scal = scal)
    Plsrstack(fm, nlv, w, Xstack, ystack, weightsstack)
end

