##
detrend(; kwargs...) = Transformer{Function, Detrend, Base.Pairs}(detrend, nothing, kwargs)
fdif(; kwargs...) = Transformer{Function, Fdif, Base.Pairs}(fdif, nothing, kwargs)
interpl(; kwargs...) = Transformer{Function, Interpl, Base.Pairs}(interpl, nothing, kwargs)
mavg(; kwargs...) = Transformer{Function, Mavg, Base.Pairs}(mavg, nothing, kwargs)
savgol(; kwargs...) = Transformer{Function, Savgol, Base.Pairs}(savgol, nothing, kwargs)
snorm(; kwargs...) = Transformer{Function, Snorm, Base.Pairs}(snorm, nothing, kwargs)
snv(; kwargs...) = Transformer{Function, Snv, Base.Pairs}(snv, nothing, kwargs)
center(; kwargs...) = Transformer{Function, Center, Base.Pairs}(center, nothing, kwargs)
scale(; kwargs...) = Transformer{Function, Scale, Base.Pairs}(scale, nothing, kwargs)
cscale(; kwargs...) = Transformer{Function, Cscale, Base.Pairs}(cscale, nothing, kwargs)
blockscal(; kwargs...) = Transformer{Function, Blockscal, Base.Pairs}(blockscal, nothing, kwargs)
##
rmgap(; kwargs...) = Transformer{Function, Rmgap, Base.Pairs}(rmgap, nothing, kwargs)
calds(; kwargs...) = Predictor{Function, CalDs, Base.Pairs}(calds, nothing, kwargs)
calpds(; kwargs...) = Predictor{Function, CalPds, Base.Pairs}(calpds, nothing, kwargs)
##
pcasvd(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcasvd, nothing, kwargs)
pcaeigen(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcaeigen, nothing, kwargs)
pcaeigenk(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcaeigenk, nothing, kwargs)
pcanipals(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcanipals, nothing, kwargs)
pcanipalsmiss(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcanipalsmiss, nothing, kwargs)
pcasph(; kwargs...) = Transformer{Function, Pca, Base.Pairs}(pcasph, nothing, kwargs)
spca(; kwargs...) = Transformer{Function, Spca, Base.Pairs}(spca, nothing, kwargs)
kpca(; kwargs...) = Transformer{Function, Kpca, Base.Pairs}(kpca, nothing, kwargs)
rp(; kwargs...) = Transformer{Function, Rp, Base.Pairs}(rp, nothing, kwargs)
##
cca(; kwargs...) = Transformer{Function, Cca, Base.Pairs}(cca, nothing, kwargs)
ccawold(; kwargs...) = Transformer{Function, Ccawold, Base.Pairs}(ccawold, nothing, kwargs)
plscan(; kwargs...) = Transformer{Function, Plscan, Base.Pairs}(plscan, nothing, kwargs)
plstuck(; kwargs...) = Transformer{Function, Plstuck, Base.Pairs}(plstuck, nothing, kwargs)
rasvd(; kwargs...) = Transformer{Function, Rasvd, Base.Pairs}(rasvd, nothing, kwargs)
##
mbconcat(; kwargs...) = Transformer{Function, Mbconcat, Base.Pairs}(mbconcat, nothing, kwargs)
mbpca(; kwargs...) = Transformer{Function, Mbpca, Base.Pairs}(mbpca, nothing, kwargs)
comdim(; kwargs...) = Transformer{Function, Comdim, Base.Pairs}(comdim, nothing, kwargs)
## Future: Could be a type TransformerXY
fda(; kwargs...) = Predictor{Function, Fda, Base.Pairs}(fda, nothing, kwargs)
fdasvd(; kwargs...) = Predictor{Function, Fda, Base.Pairs}(fdasvd, nothing, kwargs)
##
mlr(; kwargs...) = Predictor{Function, Mlr, Base.Pairs}(mlr, nothing, kwargs)
mlrchol(; kwargs...) = Predictor{Function, MlrNoArg, Base.Pairs}(mlrchol, nothing, kwargs)
mlrpinv(; kwargs...) = Predictor{Function, Mlr, Base.Pairs}(mlrpinv, nothing, kwargs)
mlrpinvn(; kwargs...) = Predictor{Function, MlrNoArg, Base.Pairs}(mlrpinvn, nothing, kwargs)
mlrvec(; kwargs...) = Predictor{Function, Mlr, Base.Pairs}(mlrvec, nothing, kwargs)
##
plskern(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plskern, nothing, kwargs)
plsnipals(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plsnipals, nothing, kwargs)
plswold(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plswold, nothing, kwargs)
plsrosa(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plsrosa, nothing, kwargs)
plssimp(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plssimp, nothing, kwargs)
cglsr(; kwargs...) = Predictor{Function, Cglsr, Base.Pairs}(cglsr, nothing, kwargs)
pcr(; kwargs...) = Predictor{Function, Pcr, Base.Pairs}(pcr, nothing, kwargs)
rrr(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(rrr, nothing, kwargs)
splskern(; kwargs...) = Predictor{Function, Splsr, Base.Pairs}(splskern, nothing, kwargs)
plsravg(; kwargs...) = Predictor{Function, Plsravg, Base.Pairs}(plsravg, nothing, kwargs)
kplsr(; kwargs...) = Predictor{Function, Kplsr, Base.Pairs}(kplsr, nothing, kwargs)
dkplsr(; kwargs...) = Predictor{Function, Dkplsr, Base.Pairs}(dkplsr, nothing, kwargs)
rr(; kwargs...) = Predictor{Function, Rr, Base.Pairs}(rr, nothing, kwargs)
rrchol(; kwargs...) = Predictor{Function, Mlr, Base.Pairs}(rrchol, nothing, kwargs)
krr(; kwargs...) = Predictor{Function, Krr, Base.Pairs}(krr, nothing, kwargs)
## 
knnr(; kwargs...) = Predictor{Function, Knnr, Base.Pairs}(knnr, nothing, kwargs)
lwmlr(; kwargs...) = Predictor{Function, Lwmlr, Base.Pairs}(lwmlr, nothing, kwargs)
lwplsr(; kwargs...) = Predictor{Function, Lwplsr, Base.Pairs}(lwplsr, nothing, kwargs)
lwplsravg(; kwargs...) = Predictor{Function, LwplsrAvg, Base.Pairs}(lwplsravg, nothing, kwargs)
##
svmr(; kwargs...) = Predictor{Function, Svmr, Base.Pairs}(svmr, nothing, kwargs)
treer_dt(; kwargs...) = Predictor{Function, TreerDt, Base.Pairs}(treer_dt, nothing, kwargs)
rfr_dt(; kwargs...) = Predictor{Function, TreerDt, Base.Pairs}(rfr_dt, nothing, kwargs)
##
mbplsr(; kwargs...) = Predictor{Function, Mbplsr, Base.Pairs}(mbplsr, nothing, kwargs)
mbplswest(; kwargs...) = Predictor{Function, Mbplswest, Base.Pairs}(mbplswest, nothing, kwargs)
rosaplsr(; kwargs...) = Predictor{Function, Rosaplsr, Base.Pairs}(rosaplsr, nothing, kwargs)
soplsr(; kwargs...) = Predictor{Function, Soplsr, Base.Pairs}(soplsr, nothing, kwargs)
## 
mlrda(; kwargs...) = Predictor{Function, Mlrda, Base.Pairs}(mlrda, nothing, kwargs)
plsrda(; kwargs...) = Predictor{Function, Plsrda, Base.Pairs}(plsrda, nothing, kwargs)
rrda(; kwargs...) = Predictor{Function, Rrda, Base.Pairs}(rrda, nothing, kwargs)
splsrda(; kwargs...) = Predictor{Function, Plsrda, Base.Pairs}(splsrda, nothing, kwargs)
kplsrda(; kwargs...) = Predictor{Function, Plsrda, Base.Pairs}(kplsrda, nothing, kwargs)
dkplsrda(; kwargs...) = Predictor{Function, Dkplsrda, Base.Pairs}(dkplsrda, nothing, kwargs)
krrda(; kwargs...) = Predictor{Function, Rrda, Base.Pairs}(krrda, nothing, kwargs)
##
lda(; kwargs...) = Predictor{Function, Lda, Base.Pairs}(lda, nothing, kwargs)
qda(; kwargs...) = Predictor{Function, Qda, Base.Pairs}(qda, nothing, kwargs)
rda(; kwargs...) = Predictor{Function, Rda, Base.Pairs}(rda, nothing, kwargs)
kdeda(; kwargs...) = Predictor{Function, Kdeda, Base.Pairs}(kdeda, nothing, kwargs)
plslda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(plslda, nothing, kwargs)
plsqda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(plsqda, nothing, kwargs)
plskdeda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(plskdeda, nothing, kwargs)
kplslda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(kplslda, nothing, kwargs)
kplsqda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(kplsqda, nothing, kwargs)
kplskdeda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(kplskdeda, nothing, kwargs)
##
splslda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(splslda, nothing, kwargs)
splsqda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(splsqda, nothing, kwargs)
splskdeda(; kwargs...) = Predictor{Function, Plslda, Base.Pairs}(splskdeda, nothing, kwargs)
##
knnda(; kwargs...) = Predictor{Function, Knnda, Base.Pairs}(knnda, nothing, kwargs)
lwmlrda(; kwargs...) = Predictor{Function, Lwmlrda, Base.Pairs}(lwmlrda, nothing, kwargs)
lwplsrda(; kwargs...) = Predictor{Function, Lwplsrda, Base.Pairs}(lwplsrda, nothing, kwargs)
lwplslda(; kwargs...) = Predictor{Function, Lwplslda, Base.Pairs}(lwplslda, nothing, kwargs)
lwplsqda(; kwargs...) = Predictor{Function, Lwplsqda, Base.Pairs}(lwplsqda, nothing, kwargs)
##
svmda(; kwargs...) = Predictor{Function, Svmda, Base.Pairs}(svmda, nothing, kwargs)
treeda_dt(; kwargs...) = Predictor{Function, TreedaDt, Base.Pairs}(treeda_dt, nothing, kwargs)
rfda_dt(; kwargs...) = Predictor{Function, TreedaDt, Base.Pairs}(rfda_dt, nothing, kwargs)
## 
mbplsrda(; kwargs...) = Predictor{Function, Mbplsrda, Base.Pairs}(mbplsrda, nothing, kwargs)
mbplslda(; kwargs...) = Predictor{Function, Mbplslda, Base.Pairs}(mbplslda, nothing, kwargs)
mbplsqda(; kwargs...) = Predictor{Function, Mbplslda, Base.Pairs}(mbplsqda, nothing, kwargs)
mbplskdeda(; kwargs...) = Predictor{Function, Mbplslda, Base.Pairs}(mbplskdeda, nothing, kwargs)
##
occstah(; kwargs...) = PredictorNoY{Function, Occstah, Base.Pairs}(occstah, nothing, kwargs)
occsd(; kwargs...) = PredictorNoY{Function, Occsd, Base.Pairs}(occsd, nothing, kwargs)
occod(; kwargs...) = Predictor{Function, Occod, Base.Pairs}(occod, nothing, kwargs)
occsdod(; kwargs...) = Predictor{Function, Occsdod, Base.Pairs}(occsdod, nothing, kwargs)


