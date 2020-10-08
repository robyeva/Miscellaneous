Consequences of school construction in Indonesia on schooling and
earnings
================
Roberta Evangelista
6/12/2020

#### Statistical analysis of data from Duflo E., 2001, “Schooling and Labor Market Consequences of School Construction in Indonesia: Evidence from an Unusual Policy Experiment”.

``` r
suppressWarnings(library(dummies))
suppressWarnings(library(AER))
```

**The data refers to the INPRES school construction program studied in
Duflo (2001) in “Schooling and Labor Market Consequences of School
Construction in Indonesia: Evidence from an Unusual Policy Experiment”.
The study analyzes whether the construction of schools in selected
regions in Indonesia affects the years of schooling and the earnings of
the treated areas.**

``` r
data = readr:::read_csv('inpres_data_corrected.csv')
```

Variables:

  - education: years of education,
  - birth\_year: year of birth,
  - birth\_region: region of birth, encoded,
  - log\_wage = logarithm of monthly earnings,
  - num\_schools = average number of schools,
  - high\_intensity: if 1, indicates treated group,
  - children71 = number of children in the district in 1971

<!-- end list -->

``` r
head(data, 6)
```

    ## # A tibble: 6 x 7
    ##   education birth_year birth_region log_wage num_schools high_intensity
    ##       <dbl>      <dbl>        <dbl>    <dbl>       <dbl>          <dbl>
    ## 1         5         71         1101     12.0        2.73              1
    ## 2         6         59         1101     11.9        2.73              1
    ## 3        11         54         1101     12.8        2.73              1
    ## 4        17         54         1101     12.8        2.73              1
    ## 5         9         68         1101     12.8        2.73              1
    ## 6         8         56         1101     13.0        2.73              1
    ## # ... with 1 more variable: children71 <dbl>

#### Regression of log monthly earnings on education

An extra year of education is estimated to impact the log monthly
earnings by 0.077 (not causal\!)

``` r
summary(lm(log_wage~education, data = data))
```

    ## 
    ## Call:
    ## lm(formula = log_wage ~ education, data = data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.5021 -0.3444  0.0533  0.3526  4.3300 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 11.402893   0.007123    1601   <2e-16 ***
    ## education    0.077033   0.000700     110   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.6011 on 45622 degrees of freedom
    ## Multiple R-squared:  0.2098, Adjusted R-squared:  0.2097 
    ## F-statistic: 1.211e+04 on 1 and 45622 DF,  p-value: < 2.2e-16

#### 1 - Generate **difference-in-difference** table to find the effect of the policy on education

``` 
               | Treated | Control | Difference
Young (>1968)  | A       | B       | C
Old (<1968)    | D       | E       | F
Difference     | G       | H       | I
```

I = difference-in-difference

First, we fill the table with the average education level for each
group. Difference-in-difference suggests that the program increases
education by 0.07 years per person on average (assuming parallel trends
for education in treated and control areas)

``` r
# People born after 1968 were affected by the INPRES school construction program. Create dummy variable for it
data$dummy_age = ifelse(data$birth_year>= 68, 1, 0)

ed_A = mean(data[(data$dummy_age==1)&(data$high_intensity==1), ]$education, na.rm=T)
ed_B = mean(data[(data$dummy_age==1)&(data$high_intensity==0), ]$education, na.rm=T)
ed_C = ed_A - ed_B
ed_D = mean(data[(data$dummy_age==0)&(data$high_intensity==1), ]$education, na.rm=T)
ed_E = mean(data[(data$dummy_age==0)&(data$high_intensity==0), ]$education, na.rm=T)
ed_F = ed_D - ed_E
ed_G = ed_A - ed_D
ed_H = ed_B - ed_E
ed_I = ed_G - ed_H

print(paste(ed_A, ed_B, ed_C)) 
```

    ## [1] "8.93788139571272 10.1183917477435 -1.1805103520308"

``` r
print(paste(ed_D, ed_E, ed_F))
```

    ## [1] "8.47585713149312 9.73272296067509 -1.25686582918197"

``` r
print(paste(ed_G, ed_H, ed_I))
```

    ## [1] "0.4620242642196 0.385668787068436 0.0763554771511643"

#### 2 - Generate difference-in-difference table to find the effect of the policy on earnings

Fill the table above using the log monthly earnings. The estimated
effect of the program on log earnings is 0.001

``` r
ed_A = mean(data[(data$dummy_age==1)&(data$high_intensity==1), ]$log_wage, na.rm=T)
ed_B = mean(data[(data$dummy_age==1)&(data$high_intensity==0), ]$log_wage, na.rm=T)
ed_C = ed_A - ed_B
ed_D = mean(data[(data$dummy_age==0)&(data$high_intensity==1), ]$log_wage, na.rm=T)
ed_E = mean(data[(data$dummy_age==0)&(data$high_intensity==0), ]$log_wage, na.rm=T)
ed_F = ed_D - ed_E
ed_G = ed_A - ed_D
ed_H = ed_B - ed_E
ed_I = ed_G - ed_H

print(paste(ed_A, ed_B, ed_C)) 
```

    ## [1] "11.8394797343139 11.9748114013597 -0.135331667045866"

``` r
print(paste(ed_D, ed_E, ed_F))
```

    ## [1] "12.1424866405218 12.2789907419889 -0.136504101467022"

``` r
print(paste(ed_G, ed_H, ed_I))
```

    ## [1] "-0.303006906207957 -0.304179340629112 0.00117243442115544"

#### 3 - Calculate the **Wald estimate** via difference-in-difference

Assumptions: the program has a causal effect on education (see above) +
exclusion restriction criterion (i.e. the program is only related to
earnings via education)

``` r
# a) - run DD regression for estimating the impact of the program on education
data$didage = data$dummy_age * data$high_intensity

ddedu = lm(education ~ high_intensity + dummy_age + didage, data = data)
summary(ddedu)
```

    ## 
    ## Call:
    ## lm(formula = education ~ high_intensity + dummy_age + didage, 
    ##     data = data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -10.1184  -3.4759   0.5241   2.2673  10.5241 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     9.73272    0.02948 330.099  < 2e-16 ***
    ## high_intensity -1.25687    0.04608 -27.277  < 2e-16 ***
    ## dummy_age       0.38567    0.05212   7.399 1.39e-13 ***
    ## didage          0.07636    0.08023   0.952    0.341    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.97 on 45620 degrees of freedom
    ## Multiple R-squared:  0.02493,    Adjusted R-squared:  0.02487 
    ## F-statistic: 388.8 on 3 and 45620 DF,  p-value: < 2.2e-16

``` r
# b) - run DD regression for estimating the impact of the program on earnings
ddearn = lm(log_wage ~ high_intensity + dummy_age + didage, data = data)
summary(ddearn)
```

    ## 
    ## Call:
    ## lm(formula = log_wage ~ high_intensity + dummy_age + didage, 
    ##     data = data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.0687 -0.3606  0.0569  0.4001  4.2786 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error  t value Pr(>|t|)    
    ## (Intercept)    12.278991   0.004882 2515.116   <2e-16 ***
    ## high_intensity -0.136504   0.007630  -17.891   <2e-16 ***
    ## dummy_age      -0.304179   0.008631  -35.243   <2e-16 ***
    ## didage          0.001172   0.013285    0.088     0.93    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.6574 on 45620 degrees of freedom
    ## Multiple R-squared:  0.05498,    Adjusted R-squared:  0.05492 
    ## F-statistic: 884.8 on 3 and 45620 DF,  p-value: < 2.2e-16

``` r
# c) - Wald estimate of the effect of education on log earnings, using INPRES exposure as the instrument
wald_estimate = ddearn$coefficients[[4]] / ddedu$coefficients[[4]]
wald_estimate
```

    ## [1] 0.01535495

#### 4 - Use the **IV regression** to compute the Wald estimate.

The coefficient of the education variable is the Wald estimate

``` r
# the interaction term is the instrument
# this is a compact form to write two regression, where the coef of education here is the Wald Estimate
fm = ivreg(log_wage ~ high_intensity + dummy_age + education | high_intensity + dummy_age + didage,
  data = data)
summary(fm)
```

    ## 
    ## Call:
    ## ivreg(formula = log_wage ~ high_intensity + dummy_age + education | 
    ##     high_intensity + dummy_age + didage, data = data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -3.01133 -0.33435  0.03416  0.38074  4.22184 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)    12.12955    1.62348   7.471 8.08e-14 ***
    ## high_intensity -0.11720    0.20576  -0.570    0.569    
    ## dummy_age      -0.31010    0.07006  -4.426 9.63e-06 ***
    ## education       0.01535    0.16698   0.092    0.927    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.6309 on 45620 degrees of freedom
    ## Multiple R-Squared: 0.1295,  Adjusted R-squared: 0.1295 
    ## Wald test: 960.5 on 3 and 45620 DF,  p-value: < 2.2e-16

#### Run an IV regression with multiple instruments and controls.

The coefficient for education suggests that an extra year of schooling
increases earnings by 0.076 log points (similar as above, but
controlling for possible bias sources).

``` r
# dummy variable for year of birth
dummy_birth_year = dummy(data$birth_year)

# dummy variable for district of birth 
dummy_region = dummy(data$birth_region)

df_iv = data.frame(dummy_region, dummy_birth_year)
df_iv$log_wage <- data$log_wage
df_iv$education <- data$education
df_iv$dummy_exposed_schools <- data$dummy_age * data$num_schools
df_iv$dummy_exposed_children <- data$children71 * data$dummy_age
```

``` r
ivregd = ivreg(log_wage~.-dummy_exposed_schools|.-education,data=df_iv)
ivregd$coefficients[which(names(ivregd$coefficients)=='education')]
```

    ##  education 
    ## 0.07578709
