residreg(pred, Y) = Y - pred

residcla(pred, y) = pred .!= y

function ssr(pred, Y)
    r = residreg(pred, Y)
    sum(r.^2, dims = 1)
end

function msep(pred, Y)
    r = residreg(pred, Y)
    reshape(colmeans(r.^2), 1, :)
end

rmsep(pred, Y) = sqrt.(msep(pred, Y))

function bias(pred, Y)
    r = residreg(pred, Y)
    reshape(colmeans(r), 1, :)
end

sep(pred, Y) = sqrt.(msep(pred, Y) .- bias(pred, Y).^2)

function r2(pred, Y)
    m = size(Y, 1)
    mu = colmeans(Y)
    zmu = reduce(hcat, fill(mu, m, 1))'
    1 .- msep(pred, Y) ./ msep(zmu, Y)
end

function cor2(pred, Y)
    q = size(Y, 2)
    res = cor(pred, Y).^2
    q == 1 ? nothing : res = diag(res)
    res = reshape(res, 1, :)
end

rpd(pred, Y) = std(Y, dims = 1) ./ sep(pred, Y) 

rpq(pred, Y) = mapslices(Jchemo.iqr, Y, dims = 1) ./ sep(pred, Y) 

function err(pred, y)
    r = residcla(pred, y)
    sum(r) / size(y, 1)
end

function mse(pred, Y)
    q = size(Y, 2)
    zmsep = msep(pred, Y)
    zrmsep = sqrt.(zmsep)
    zsep = sep(pred, Y)
    zbias = bias(pred, Y)
    zr2 = r2(pred, Y)
    zcor2 = cor2(pred, Y)
    zrpd = rpd(pred, Y)
    zrpq = rpq(pred, Y)
    zmean = reshape(colmeans(Y), 1, :)
    nam = map(string, repeat(["y"], q), 1:q)
    nam = reshape(nam, 1, :)
    res = (nam = nam, msep = zmsep, rmsep = zrmsep, sep = zsep, bias = zbias, 
        cor2 = zcor2, r2 = zr2, rpd = zrpd, rpq = zrpq, mean = zmean)
    DataFrame(res)
end


