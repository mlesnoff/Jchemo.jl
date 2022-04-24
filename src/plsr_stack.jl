function plsr_stack(X, y, weights = ones(size(X, 1)); nlv, K = 5, rep = 10)
    plsr_stack!(copy(ensure_mat(X)), copy(ensure_mat(y)), weights; nlv = nlv, K = K, rep = rep)
end

function plsr_stack!(X::Matrix, y::Matrix, weights = ones(size(X, 1)); nlv, K = 5, rep = 10)
    n, p = size(X)
    weights = mweight(weights)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    segm = segmkf(n, K; rep = rep)
    ystack = []
    Xstack = []
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
            zfm = plskern(Xcal, ycal, wcal; nlv =  nlvmax) ;
            pred = Jchemo.predict(zfm, Xval; nlv = 0:nlvmax).pred
            pred = reduce(hcat, pred)
            if k == 1
                Xstack = copy(pred) 
                ystack = copy(yval)
                weights_stack = copy(wval)
            else
                Xstack = vcat(Xstack, pred) 
                ystack = vcat(ystack, yval)
                weights_stack = vcat(weights_stack, wval)
            end
            k = k + 1
        end
    end
    weights_stack = mweight(weights_stack)
    Xstack = vcol(Xstack, nlv .+ 1)
    XtD = Xstack' * Diagonal(weights_stack)
    w = vec(cholesky!(Hermitian(XtD * Xstack)) \ (XtD * ystack))
    fm = plskern!(X, y, weights; nlv = nlvmax)
    PlsrStack(fm, nlv, w, Xstack, ystack, weights_stack)
end

