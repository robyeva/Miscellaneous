Long-lasting mining Mita effects on household consumption
================
Roberta Evangelista
8/7/2020

### Analysis of data from Dell (2010), The Persistent Effects of Peru’s Mining Mita

``` r
suppressWarnings(library(lmtest))
suppressWarnings(library(sandwich))
suppressWarnings(library(dplyr))
```

#### Data extraction

**Data refers to the paper: Dell (2010), “The Persistent Effects of
Peru’s Mining Mita”, which analyzes whether a forced labor institution
(the “Mita”, set up by the Spanish empire in Peru and Bolivia in 16th
century and lasting until \~200y ago) still has effects on household
consumption and income today**

``` r
data = readr:::read_csv('mitaData_corrected.csv')
```

Relevant variables are:

  - lhhequiv: log equivalent household consumption
  - elv\_sh: elevation
  - pothuan\_mita: if 1, the household is inside the Mita area (treated)
  - x,y = longitude and latitude variables
  - dpot: distance from Potosi (large city in colonial times)
  - d\_bnd: distance from the Mita boundary
  - slope: mean slope
  - infants, children, adults: numbers per household
  - bfe4\_1, bfe4\_2, bfe4\_3: boundary segment fixed effects
    (inside/outside the Mita)

#### Create polynomial variables for latitude and longitude

``` r
data = mutate(data, x_2 = x * x, 
               y_2 = y * y,
               xy = x * y,
               x_3 = x * x * x,
               y_3 = y * y * y,
               x2_y = x * x * y,
               x_y2 = x * y * y)
```

#### Filter data by distance to the border

``` r
data_100km = filter(data, d_bnd <= 100) 
data_75km = filter(data, d_bnd <= 75)
data_50km = filter(data, d_bnd <= 50)
```

#### Regress the log equivalent household consumption on relevant variables (see above) for the households at up to 100 km from the border

The coefficient estimate for Mita is -0.284, suggesting that belonging
to the Mita district is associated with a 28% decrease in log household
consumption (for households in \<100 km from the border).

``` r
reg100 = lm(lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + adults + bfe4_1 + bfe4_2 + bfe4_3, data=data_100km)
summary(reg100)
```

    ## 
    ## Call:
    ## lm(formula = lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + 
    ##     x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + 
    ##     adults + bfe4_1 + bfe4_2 + bfe4_3, data = data_100km)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6.2697 -0.4169  0.0565  0.5374  2.7097 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   5.396668   0.553339   9.753  < 2e-16 ***
    ## pothuan_mita -0.284115   0.122321  -2.323 0.020332 *  
    ## x            -0.072379   0.153573  -0.471 0.637498    
    ## y            -1.006262   0.304276  -3.307 0.000966 ***
    ## x_2           0.055537   0.067377   0.824 0.409918    
    ## y_2           0.071916   0.138282   0.520 0.603095    
    ## xy            0.049335   0.129439   0.381 0.703153    
    ## x_3           0.010049   0.056632   0.177 0.859181    
    ## y_3           0.597786   0.162699   3.674 0.000247 ***
    ## x2_y          0.140899   0.126097   1.117 0.264013    
    ## x_y2          0.293298   0.161222   1.819 0.069083 .  
    ## elv_sh        0.067702   0.115813   0.585 0.558920    
    ## slope        -0.022199   0.012335  -1.800 0.072129 .  
    ## infants      -0.004066   0.036274  -0.112 0.910775    
    ## children      0.013635   0.020479   0.666 0.505652    
    ## adults        0.015014   0.021103   0.711 0.476920    
    ## bfe4_1        0.878131   0.342565   2.563 0.010465 *  
    ## bfe4_2        0.601801   0.226655   2.655 0.008014 ** 
    ## bfe4_3        0.090882   0.155872   0.583 0.559946    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9856 on 1459 degrees of freedom
    ## Multiple R-squared:  0.05946,    Adjusted R-squared:  0.04785 
    ## F-statistic: 5.124 on 18 and 1459 DF,  p-value: 1.491e-11

#### Helper functions to cluster the standard errors by district

``` r
#1 - calculate the variance covariance matrix
get_CL_vcov <- function(model, cluster){
M = length(unique(cluster))
N = length(cluster)
K = model$rank
dfc = (M/(M-1)) * ((N-1)/(N-K))
uj = apply(estfun(model), 2, function(x) tapply(x,cluster,sum))
vcovCL = dfc * sandwich(model, meat=crossprod(uj)/N)
return(vcovCL)
}

# 2- Calculate the number of degrees of freedom
get_CL_df = function(model, cluster){
M = length(unique(cluster)) 
df = M-1
return(df)
}
```

#### Cluster the stanard error by district to evaluate significance

The p-value for the Mita variable shows that the result is not
significant

``` r
reg100_covmat = get_CL_vcov(reg100, data_100km$district)
reg100_degree_freedom = get_CL_df(reg100, data_100km$district)
coeftest(reg100, reg100_covmat, df=reg100_degree_freedom)
```

    ## 
    ## t test of coefficients:
    ## 
    ##                Estimate Std. Error t value  Pr(>|t|)    
    ## (Intercept)   5.3966678  0.7767630  6.9476 1.556e-09 ***
    ## pothuan_mita -0.2841147  0.1988987 -1.4284  0.157612    
    ## x            -0.0723788  0.1760388 -0.4112  0.682217    
    ## y            -1.0062619  0.4607141 -2.1841  0.032305 *  
    ## x_2           0.0555372  0.0712303  0.7797  0.438205    
    ## y_2           0.0719158  0.1945532  0.3696  0.712762    
    ## xy            0.0493348  0.1525006  0.3235  0.747277    
    ## x_3           0.0100493  0.0682042  0.1473  0.883286    
    ## y_3           0.5977860  0.2755892  2.1691  0.033472 *  
    ## x2_y          0.1408986  0.1491712  0.9445  0.348141    
    ## x_y2          0.2932984  0.1956509  1.4991  0.138347    
    ## elv_sh        0.0677018  0.1584490  0.4273  0.670488    
    ## slope        -0.0221990  0.0167926 -1.3219  0.190489    
    ## infants      -0.0040656  0.0259563 -0.1566  0.875986    
    ## children      0.0136345  0.0158243  0.8616  0.391840    
    ## adults        0.0150135  0.0230997  0.6499  0.517856    
    ## bfe4_1        0.8781309  0.2888836  3.0397  0.003329 ** 
    ## bfe4_2        0.6018009  0.3884754  1.5491  0.125859    
    ## bfe4_3        0.0908820  0.1805537  0.5034  0.616298    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#### Same regression as above, but with a 75 km cutoff

Being in a Mita district is associated with a 21% decrease in household
consumption (not significant).

``` r
reg75 = lm(lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + adults + bfe4_1 + bfe4_2 + bfe4_3, data=data_75km)
summary(reg75)
```

    ## 
    ## Call:
    ## lm(formula = lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + 
    ##     x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + 
    ##     adults + bfe4_1 + bfe4_2 + bfe4_3, data = data_75km)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6.0252 -0.4025  0.0127  0.4548  2.6109 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   5.593197   0.659524   8.481  < 2e-16 ***
    ## pothuan_mita -0.216447   0.124278  -1.742 0.081842 .  
    ## x             0.005405   0.180542   0.030 0.976122    
    ## y            -0.992426   0.328133  -3.024 0.002546 ** 
    ## x_2          -0.018506   0.090745  -0.204 0.838439    
    ## y_2           0.487267   0.199743   2.439 0.014860 *  
    ## xy            0.306612   0.147059   2.085 0.037295 *  
    ## x_3          -0.226263   0.113499  -1.994 0.046441 *  
    ## y_3           0.906216   0.214679   4.221 2.62e-05 ***
    ## x2_y         -0.278396   0.164020  -1.697 0.089908 .  
    ## x_y2          0.543332   0.161010   3.375 0.000764 ***
    ## elv_sh       -0.015936   0.137159  -0.116 0.907524    
    ## slope        -0.032520   0.013838  -2.350 0.018941 *  
    ## infants      -0.008816   0.036939  -0.239 0.811409    
    ## children     -0.005231   0.020486  -0.255 0.798506    
    ## adults        0.025081   0.023065   1.087 0.277101    
    ## bfe4_1        0.932777   0.328556   2.839 0.004605 ** 
    ## bfe4_2        0.901669   0.250842   3.595 0.000339 ***
    ## bfe4_3        0.172509   0.150655   1.145 0.252426    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.8943 on 1142 degrees of freedom
    ## Multiple R-squared:  0.06005,    Adjusted R-squared:  0.04524 
    ## F-statistic: 4.054 on 18 and 1142 DF,  p-value: 2.807e-08

``` r
reg75_covmat = get_CL_vcov(reg75, data_75km$district)
reg75_degree_freedom = get_CL_df(reg75, data_75km$district)
coeftest(reg75, reg75_covmat, df=reg75_degree_freedom)
```

    ## 
    ## t test of coefficients:
    ## 
    ##               Estimate Std. Error t value  Pr(>|t|)    
    ## (Intercept)   5.593197   0.662996  8.4363 1.005e-11 ***
    ## pothuan_mita -0.216447   0.207492 -1.0432 0.3011305    
    ## x             0.005405   0.196206  0.0275 0.9781159    
    ## y            -0.992426   0.384100 -2.5838 0.0122692 *  
    ## x_2          -0.018506   0.105331 -0.1757 0.8611342    
    ## y_2           0.487267   0.256159  1.9022 0.0620292 .  
    ## xy            0.306612   0.158368  1.9361 0.0576543 .  
    ## x_3          -0.226263   0.139115 -1.6264 0.1091852    
    ## y_3           0.906216   0.275565  3.2886 0.0017006 ** 
    ## x2_y         -0.278396   0.194940 -1.4281 0.1585319    
    ## x_y2          0.543332   0.196579  2.7639 0.0076072 ** 
    ## elv_sh       -0.015936   0.141337 -0.1128 0.9106087    
    ## slope        -0.032520   0.016340 -1.9903 0.0512014 .  
    ## infants      -0.008816   0.030190 -0.2920 0.7713018    
    ## children     -0.005231   0.016456 -0.3179 0.7517087    
    ## adults        0.025080   0.029877  0.8395 0.4045966    
    ## bfe4_1        0.932777   0.263924  3.5343 0.0008023 ***
    ## bfe4_2        0.901669   0.301076  2.9948 0.0040091 ** 
    ## bfe4_3        0.172509   0.160448  1.0752 0.2866761    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#### Same regression as above, but with a 50 km cutoff

Being in a Mita district is associated with a 33% decrease in household
consumption (not significant)

``` r
reg50 = lm(lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + adults + bfe4_1 + bfe4_2 + bfe4_3, data=data_50km)
summary(reg50)
```

    ## 
    ## Call:
    ## lm(formula = lhhequiv ~ pothuan_mita + x + y + x_2 + y_2 + xy + 
    ##     x_3 + y_3 + x2_y + x_y2 + elv_sh + slope + infants + children + 
    ##     adults + bfe4_1 + bfe4_2 + bfe4_3, data = data_50km)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6.0733 -0.3895  0.0014  0.4367  2.4120 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   5.353195   0.648257   8.258 4.69e-16 ***
    ## pothuan_mita -0.331078   0.125897  -2.630  0.00868 ** 
    ## x            -0.135645   0.263134  -0.515  0.60632    
    ## y            -0.747280   0.381273  -1.960  0.05028 .  
    ## x_2           0.238377   0.119043   2.002  0.04551 *  
    ## y_2           0.286559   0.245654   1.167  0.24369    
    ## xy            0.176397   0.229761   0.768  0.44282    
    ## x_3          -0.152184   0.191978  -0.793  0.42813    
    ## y_3           0.633272   0.230229   2.751  0.00606 ** 
    ## x2_y         -0.196753   0.199511  -0.986  0.32429    
    ## x_y2          0.448451   0.213135   2.104  0.03562 *  
    ## elv_sh        0.040463   0.134988   0.300  0.76443    
    ## slope        -0.014059   0.014281  -0.985  0.32511    
    ## infants      -0.035844   0.036874  -0.972  0.33126    
    ## children     -0.007878   0.020352  -0.387  0.69879    
    ## adults        0.010570   0.022986   0.460  0.64572    
    ## bfe4_1        0.839856   0.467855   1.795  0.07294 .  
    ## bfe4_2        0.620983   0.280695   2.212  0.02717 *  
    ## bfe4_3        0.227754   0.158124   1.440  0.15008    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.8319 on 994 degrees of freedom
    ## Multiple R-squared:  0.06935,    Adjusted R-squared:  0.0525 
    ## F-statistic: 4.115 on 18 and 994 DF,  p-value: 2.06e-08

``` r
reg50_covmat = get_CL_vcov(reg50, data_50km$district)
reg50_degree_freedom = get_CL_df(reg50, data_50km$district)
coeftest(reg50, reg50_covmat, df=reg50_degree_freedom)
```

    ## 
    ## t test of coefficients:
    ## 
    ##                Estimate Std. Error t value  Pr(>|t|)    
    ## (Intercept)   5.3531953  0.5923430  9.0373 3.618e-12 ***
    ## pothuan_mita -0.3310776  0.2192494 -1.5100   0.13720    
    ## x            -0.1356446  0.3039319 -0.4463   0.65727    
    ## y            -0.7472798  0.4334150 -1.7242   0.09073 .  
    ## x_2           0.2383767  0.1357036  1.7566   0.08499 .  
    ## y_2           0.2865588  0.3422686  0.8372   0.40637    
    ## xy            0.1763966  0.2357436  0.7483   0.45774    
    ## x_3          -0.1521836  0.2441907 -0.6232   0.53592    
    ## y_3           0.6332718  0.2894293  2.1880   0.03328 *  
    ## x2_y         -0.1967531  0.2365914 -0.8316   0.40950    
    ## x_y2          0.4484506  0.2239959  2.0020   0.05061 .  
    ## elv_sh        0.0404628  0.1262467  0.3205   0.74989    
    ## slope        -0.0140592  0.0181847 -0.7731   0.44301    
    ## infants      -0.0358442  0.0302821 -1.1837   0.24203    
    ## children     -0.0078777  0.0178544 -0.4412   0.66092    
    ## adults        0.0105700  0.0286672  0.3687   0.71387    
    ## bfe4_1        0.8398561  0.4218180  1.9910   0.05185 .  
    ## bfe4_2        0.6209832  0.3611417  1.7195   0.09159 .  
    ## bfe4_3        0.2277537  0.1740912  1.3082   0.19666    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#### Using a different positional measure

Instead of using latitude and longitude, we can use the distance to
Potosi (dpot), which in colonial times was the largest city in the area
and has been found as “an important determinant of local production and
trading activities, and access to coinage” (Dell 2010).

#### Create polynomial variables of distance to Potosi

``` r
data$dpot2 = data$dpot * data$dpot
data$dpot3 = data$dpot * data$dpot * data$dpot
```

``` r
data_100km = filter(data, d_bnd <= 100) 
```

#### Estimate effect on household consumption (100 km cutoff) using the *dpot* variables (instead of longitude and latitude)

The estimated coefficient of formerly belonging to the Mita district is
similar to previous regressions (estimated 33% decrease in household
consumption), but is significant (the same holds for 75 km and 50 km
cutoffs, not shown). An hypothesis to explain this result is the
distance to Potosi is a more economically meaningful variable, which
captures a larger share of the residual variance of the outcome
variable.

``` r
pot100 = lm(lhhequiv ~ pothuan_mita + dpot + dpot2 + dpot3 + elv_sh + slope + infants + children + adults + bfe4_1 + bfe4_2 + bfe4_3, data=data_100km)
summary(pot100)
```

    ## 
    ## Call:
    ## lm(formula = lhhequiv ~ pothuan_mita + dpot + dpot2 + dpot3 + 
    ##     elv_sh + slope + infants + children + adults + bfe4_1 + bfe4_2 + 
    ##     bfe4_3, data = data_100km)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6.0942 -0.4359  0.0814  0.5355  2.6544 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  16.493680   8.782835   1.878   0.0606 .  
    ## pothuan_mita -0.336809   0.069074  -4.876 1.20e-06 ***
    ## dpot         -2.838344   3.213102  -0.883   0.3772    
    ## dpot2         0.269943   0.382310   0.706   0.4802    
    ## dpot3        -0.008264   0.014795  -0.559   0.5766    
    ## elv_sh       -0.175993   0.093718  -1.878   0.0606 .  
    ## slope        -0.028484   0.011811  -2.412   0.0160 *  
    ## infants      -0.010626   0.036358  -0.292   0.7701    
    ## children      0.010359   0.020452   0.507   0.6126    
    ## adults        0.016994   0.021195   0.802   0.4228    
    ## bfe4_1        0.514985   0.108651   4.740 2.35e-06 ***
    ## bfe4_2       -0.070514   0.155252  -0.454   0.6498    
    ## bfe4_3        0.083992   0.114889   0.731   0.4649    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9904 on 1465 degrees of freedom
    ## Multiple R-squared:  0.04632,    Adjusted R-squared:  0.03851 
    ## F-statistic:  5.93 on 12 and 1465 DF,  p-value: 3.612e-10

``` r
pot100_covmat = get_CL_vcov(pot100, data_100km$district)
pot100_degree_freedom = get_CL_df(pot100, data_100km$district)
coeftest(pot100, pot100_covmat, df=pot100_degree_freedom)
```

    ## 
    ## t test of coefficients:
    ## 
    ##                Estimate Std. Error t value  Pr(>|t|)    
    ## (Intercept)  16.4936803 12.8307089  1.2855 0.2028596    
    ## pothuan_mita -0.3368093  0.0870028 -3.8712 0.0002405 ***
    ## dpot         -2.8383436  4.7047423 -0.6033 0.5482629    
    ## dpot2         0.2699427  0.5655012  0.4774 0.6345991    
    ## dpot3        -0.0082637  0.0220647 -0.3745 0.7091476    
    ## elv_sh       -0.1759931  0.1184048 -1.4864 0.1416722    
    ## slope        -0.0284840  0.0169523 -1.6802 0.0973682 .  
    ## infants      -0.0106256  0.0257637 -0.4124 0.6812895    
    ## children      0.0103591  0.0167990  0.6167 0.5394642    
    ## adults        0.0169945  0.0235730  0.7209 0.4733540    
    ## bfe4_1        0.5149850  0.1174168  4.3860 3.988e-05 ***
    ## bfe4_2       -0.0705140  0.2485322 -0.2837 0.7774611    
    ## bfe4_3        0.0839921  0.1456158  0.5768 0.5659207    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Disclaimer: this notebook is inspired by exercises of the “Foundations
of Development Policy” course offered by
[edX](https://www.edx.org/course/foundations-of-development-policy)
