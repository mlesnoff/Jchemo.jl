mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)
  
X = dat.X 
Y = dat.Y
y = Y.y    
year = Y.year
tab(year)
wl = names(X)
wl_num = parse.(Float64, wl)

######## End Data

#### Spectra

plotsp(X, wl_num; nsamp = 10, 
    xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

## Preprocessing
Xp = savgol(snv(X); f = 21, pol = 3, d = 2)

plotsp(Xp, wl_num; nsamp = 10,
    xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

#### PCA

fm = pcasvd(Xp, nlv = 10) ; 
pnames(fm)

res = summary(fm, Xp) ;
pnames(res)
z = res.explvarx
plotgrid(z.lv, 100 * z.pvar; step = 1,
    xlabel = "nb. PCs", ylabel = "% variance explained").f

T = fm.T
plotxy(T[:, 1], T[:, 2]; color = (:red, .5),
    xlabel = "PC1", ylabel = "PC2").f

plotxy(T[:, 1], T[:, 2], year; ellipse = true,
    xlabel = "PC1", ylabel = "PC2").f

######## Variable y

summ(y)

f = Figure(resolution = (500, 400))
Axis(f[1, 1], xlabel = "TBC", ylabel = "Nb. samples")
hist!(y; bins = 50)
f

f = Figure(resolution = (500, 400))
Axis(f[1, 1], xlabel = "Year", ylabel = "TBC")
boxplot!(year, y; show_notch = true)
f


