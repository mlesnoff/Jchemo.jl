module Jchemo  # Start-Module

using Clustering
using DataInterpolations
using DecisionTree
using Distributions
using DataFrames
using Distances
using ImageFiltering     # convolutions in preprocessing (mavg, savgol)
using Interpolations
using LIBSVM 
using LinearAlgebra
using Makie
using NearestNeighbors
using Random
using SparseArrays 
using Statistics
using StatsBase          # sample

include("_structures_param.jl")
include("_structures_fit.jl")
include("_structures_mod_stru.jl")
include("_structures_mod_fun.jl")
include("_structures_mod.jl")
include("_structures_mod_pipelines.jl")

###### Misc
include("utility.jl")
include("utility_colwise.jl")
include("utility_rowwise.jl")
include("utility_cscale.jl")
include("utility_blockscal.jl")
include("colmedspa.jl")
include("fweight.jl") 
include("ellipse.jl")
include("matW.jl")
include("nipals.jl")
include("nipalsmiss.jl")
include("preprocessing.jl") 
include("rmgap.jl")
include("snipals.jl")
include("snipalsh.jl")
include("snipalsmix.jl")

###### Graphics
include("plotsp.jl")
include("plotgrid.jl")
include("plotxy.jl")
include("plotconf.jl")

###### Distributions
include("dmnorm.jl")
include("dmnormlog.jl")
include("dmkern.jl")

###### Exploratory
include("fda.jl")     # Here since ::Fda called in pcasvd
include("fdasvd.jl")     
include("pcasvd.jl")
include("pcaeigen.jl")
include("pcaeigenk.jl")
include("pcanipals.jl")
include("pcanipalsmiss.jl")
include("kpca.jl")
include("pcasph.jl") 
include("rpmat.jl")
include("rp.jl")

## Sparse
include("spca.jl")

## Multiblock 
include("angles.jl")
include("mblock.jl")
include("mbpca.jl")
include("comdim.jl")
include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")

###### Regression 
include("aov1.jl")
include("mlr.jl")
include("rr.jl")
include("rrchol.jl")
include("pcr.jl")
include("rrr.jl") 
include("plskern.jl")
include("plsnipals.jl")
include("plswold.jl") 
include("plsrosa.jl")
include("plssimp.jl")
include("plsravg.jl")
include("plsravg_unif.jl")
include("cglsr.jl")
include("krr.jl")
include("kplsr.jl")
include("dkplsr.jl")
include("dfplsr_cg.jl")
include("aicplsr.jl")
include("vip.jl") 

include("xfit.jl")
include("xresid.jl")

## Sparse
include("splskern.jl")

## Multiblock
include("mbplsr.jl") 
include("mbplswest.jl")
include("rosaplsr.jl") 
include("soplsr.jl") 

## Local
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwplsr.jl")
include("lwplsravg.jl")

## Validation
include("mpar.jl")
include("scores.jl")
include("confusion.jl")
include("segmkf.jl")
include("segmts.jl")
include("gridscore.jl")
include("gridscore_br.jl")
include("gridscore_lv.jl")
include("gridscore_lb.jl")
include("gridcv.jl")
include("gridcv_br.jl")
include("gridcv_lv.jl")
include("gridcv_lb.jl")
include("selwold.jl")

## Variable importance (direct methods) 
include("isel.jl")
include("viperm.jl")

## Svm, Trees
include("svmr.jl")
include("treer_dt.jl")
include("rfr_dt.jl")

###### Discrimination 
include("lda.jl")
include("qda.jl")
include("rda.jl")
include("kdeda.jl")
include("mlrda.jl")
include("rrda.jl")
include("plsrda.jl") 
include("plslda.jl")
include("plsqda.jl")
include("plskdeda.jl")
include("krrda.jl")
include("kplsrda.jl")
include("dkplsrda.jl")

## Sparse
include("splsrda.jl")
include("splslda.jl")
include("splsqda.jl")
include("splskdeda.jl")

## One-class
include("occsd.jl")
include("occod.jl") 
include("occsdod.jl")
include("occstah.jl")
include("stah.jl")

## Local
include("lwmlrda.jl")
include("lwplsrda.jl")
include("lwplslda.jl")
include("lwplsqda.jl")
include("knnda.jl")

## Svm, Trees
include("svmda.jl")
include("treeda_dt.jl")
include("rfda_dt.jl")

###### Calibration transfer
include("calds.jl")
include("calpds.jl")
include("difmean.jl")
include("eposvd.jl")

###### Sampling
include("sampks.jl")
include("sampdp.jl")
include("samprand.jl")
include("sampsys.jl")
include("sampcla.jl")
include("sampdf.jl")

include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    Par,
    Transformer, Predictor,
    fit!,
    transf!,
    pip,
    ###### Utilities
    aggstat,
    dupl, miss,
    fcenter, fcenter!, 
    colmad, colmean, colnorm, colstd, colsum, colvar,
    colmeanskip, colstdskip, colsumskip, colvarskip,
    corm, covm,
    cosm, cosv,
    fcscale, fcscale!, 
    dummy,
    ensure_df, ensure_mat,
    findmax_cla, 
    frob,
    fweight,
    head, @head,
    list, 
    matB, matW, 
    mblock,
    mlev,
    mweight,
    nco,
    nipals,
    nipalsmiss,
    normw,  
    nro,
    out,
    plist, 
    pmod, pnames, psize,
    pval,
    recodcat2int, recodnum2cla, 
    recovkwargs,
    replacebylev, replacebylev2, 
    replacedict, 
    rmcol, rmrow, 
    rowmean, rowstd, rowsum, rowvar,
    rowmeanskip, rowstdskip, rowsumskip, rowvarskip,   
    fscale, fscale!,
    snipals, snipalsh, snipalsmix,
    soft,
    softmax,
    sourcedir,
    ssq,
    summ,
    tab, tabdf, tabdupl,
    vcatdf,
    vcol, vrow,
    ###### Distributions
    dmnorm, dmnorm!,
    dmnormlog, dmnormlog!,
    dmkern,
    ## Pre-processing
    detrend, detrend!, 
    fdif, fdif!,
    interpl, 
    center, scale, cscale,
    #cubic_spline,
    mavg, mavg!, 
    rmgap, rmgap!,
    savgk, savgol, savgol!,
    snv, snv!, 
    ###### Calibration ransfer
    calds, calpds,
    difmean,
    eposvd,
    ###### Exploratory
    kpca,
    pcasvd, pcasvd!, 
    pcaeigen, pcaeigen!, 
    pcaeigenk, pcaeigenk!,
    pcanipals, pcanipals!,
    pcanipalsmiss, pcanipalsmiss!,
    rpmatgauss, rpmatli, rp, rp!,
    pcasph, pcasph!,
    spca, spca!,
    ## Multiblock
    fblockscal, fblockscal_frob, fblockscal_mfa,
    fblockscal_ncol, fblockscal_sd,
    rv, lg, rd, 
    mbpca, mbpca!,
    comdim, comdim!, 
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    ###### Regression
    aov1,
    mlr, mlr!, mlrchol, mlrchol!, 
    mlrpinv, mlrpinv!, mlrpinvn, mlrpinvn!,
    mlrvec, mlrvec!,
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    cglsr, cglsr!,
    pcr,
    rr, rr!, rrchol, rrchol!,
    rrr, rrr!,   
    krr, krr!, kplsr, kplsr!, 
    dkplsr, dkplsr!,
    plsravg, plsravg!,
    dfplsr_cg, aicplsr,
    svmr,
    treer_dt, rfr_dt, 
    ## Sparse 
    splskern, splskern!, 
    ## Multi-block
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    rosaplsr, rosaplsr!,
    soplsr,
    ## Variable selection/importance (direct methods) 
    isel!,
    vip, 
    viperm!,
    ## Utils
    xfit, xfit!, xresid, xresid!,
    ## Local
    locw, locwlv,
    knnr,
    lwmlr,
    lwplsr, lwplsravg,
    ###### Discrimination
    fda, fda!, fdasvd, fdasvd!,
    mlrda,
    rrda, krrda,
    lda, qda, kdeda,
    rda,
    plsrda,
    plslda, plsqda, plskdeda,
    kplsrda, dkplsrda,
    svmda, 
    treeda_dt, rfda_dt,
    stah,
    occstah,
    occsd, occod, occsdod,
    ## Sparse 
    splsrda,
    splslda, splsqda,
    splskdeda,
    ## Local 
    lwmlrda,
    lwplsrda, 
    lwplslda, lwplsqda,
    knnda,
    ## Auxiliary
    transf, coef, predict,
    transfbl, 
    ## Validation
    residreg, residcla, 
    ssr, msep, rmsep, rmsepstand, 
    bias, sep, cor2, r2, rpd, rpdr, mse, errp,
    mpar,
    gridscore, 
    gridscore_br, gridscore_lv, gridscore_lb,
    segmts, segmkf,
    gridcv, 
    gridcv_br, gridcv_lv, gridcv_lb, 
    selwold,
    confusion, 
    ###### Sampling
    sampks, sampdp, samprand, sampsys, sampcla, 
    sampdf,
    ###### Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol,
    ###### Graphics
    plotconf,
    plotgrid, 
    plotsp,
    plotxy
    ## Not exported since surcharge:
    ## - summary => Base.summary

end # End-Module




