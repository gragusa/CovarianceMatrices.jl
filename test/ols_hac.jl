nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
df = readtable(nf)
lm1 = glm(y~x+w, df, Normal(), IdentityLink())

V = vcov(lm1, TruncatedKernel(1.0), prewhite = false)
Vt = [0.004124066084299 -0.000125870872864  0.000353984580059;
      -0.000125870872864  0.003123789970617 -0.000376282603066;
       0.000353984580059 -0.000376282603066  0.003395555576658]
@test_approx_eq V Vt

V = vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = false)
Vt = [3.55782156072e-03 -3.96292254354e-04  5.68526125082e-05;
      -3.96292254354e-04  3.39720366644e-03 -9.21982139463e-05;
       5.68526125082e-05 -9.21982139463e-05  2.99762549492e-03]
@test_approx_eq V Vt

V = vcov(lm1, BartlettKernel(1.0), prewhite = false)
Vt = [3.47574230254e-03 -4.48633600585e-04  1.78075451032e-05;
     -4.48633600585e-04  3.44096545157e-03 -4.21085727843e-05;
      1.78075451032e-05 -4.21085727843e-05  2.92262985776e-03]
@test_approx_eq V Vt

V = vcov(lm1, ParzenKernel(1.0), prewhite = false)
Vt = [3.47574230254e-03 -4.48633600585e-04  1.78075451032e-05;
     -4.48633600585e-04  3.44096545157e-03 -4.21085727843e-05;
      1.78075451032e-05 -4.21085727843e-05  2.92262985776e-03]
@test_approx_eq V Vt

V = vcov(lm1, TukeyHanningKernel(1.0), prewhite = false)
Vt = [3.47574230254e-03 -4.48633600585e-04  1.78075451032e-05;
     -4.48633600585e-04  3.44096545157e-03 -4.21085727843e-05;
      1.78075451032e-05 -4.21085727843e-05  2.92262985776e-03]
@test_approx_eq V Vt

V = vcov(lm1, TruncatedKernel(1.0), prewhite = true)
Vt = [0.004075081761409 -0.000240203136461  0.000354669670023;
     -0.000240203136461  0.003028831540918 -0.000469930368878;
      0.000354669670023 -0.000469930368878  0.003500263698496]
@test_approx_eq V Vt

V = vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = true)
Vt = [0.00416915902941 -0.000133650201140  0.000395649595840;
     -0.00013365020114  0.003112706448531 -0.000424303480269;
      0.00039564959584 -0.000424303480269  0.003463428791423]
@test_approx_eq V Vt

V = vcov(lm1, BartlettKernel(1.0), prewhite = true)
Vt = [0.004193163452767 -0.000126916717674  0.000412982604681;
     -0.000126916717674  0.003122220514779 -0.000409693229837;
      0.000412982604681 -0.000409693229837  0.003445964499879]
@test_approx_eq V Vt

V = vcov(lm1, ParzenKernel(1.0), prewhite = true)
Vt = [0.004193163452767 -0.000126916717674  0.000412982604681;
     -0.000126916717674  0.003122220514779 -0.000409693229837;
      0.000412982604681 -0.000409693229837  0.003445964499879]
@test_approx_eq V Vt

V = vcov(lm1, TukeyHanningKernel(1.0), prewhite = true)
Vt =  [ 0.004193163452767 -0.000126916717674  0.000412982604681;
       -0.000126916717674  0.003122220514779 -0.000409693229837;
       0.000412982604681 -0.000409693229837  0.003445964499879]
@test_approx_eq V Vt


V = vcov(lm1, BartlettKernel(NeweyWest), prewhite = false)
Vt = [0.00425404   -0.000402832   0.000285714
     -0.000402832   0.00282219   -0.000261739
      0.000285714  -0.000261739   0.00306664 ]
@test_approx_eq_eps V Vt 1e-8

V = vcov(lm1, QuadraticSpectralKernel(NeweyWest), prewhite = false)
Vt = [0.00370396   -0.000265251   0.000162278
     -0.000265251   0.0033624    -0.000175805
      0.000162278  -0.000175805   0.00315178 ]
@test_approx_eq_eps V Vt 1e-8

V = vcov(lm1, ParzenKernel(NeweyWest), prewhite = false)
Vt = [0.00384234   -0.000305152   0.000189532
     -0.000305152   0.00323359   -0.000215547
      0.000189532  -0.000215547   0.00314338 ]
@test_approx_eq_eps V Vt 1e-8


@test CovarianceMatrices.bwAndrews(lm1, BartlettKernel()) ≈ 1.79655260917
@test CovarianceMatrices.bwAndrews(lm1, TruncatedKernel()) ≈ 0.923095757094
@test CovarianceMatrices.bwAndrews(lm1, ParzenKernel()) ≈ 3.71612017536
@test CovarianceMatrices.bwAndrews(lm1, QuadraticSpectralKernel()) ≈ 1.84605188391

@test CovarianceMatrices.bwAndrews(lm1, BartlettKernel(), prewhite = true) ≈ 0.547399170212
@test CovarianceMatrices.bwAndrews(lm1, TruncatedKernel(), prewhite = true) ≈ 0.422530519468
@test CovarianceMatrices.bwAndrews(lm1, ParzenKernel(), prewhite = true) ≈ 1.70098733098
@test CovarianceMatrices.bwAndrews(lm1, QuadraticSpectralKernel(), prewhite = true) ≈ 0.844997125683

@test CovarianceMatrices.bwNeweyWest(lm1, BartlettKernel(), prewhite = false) ≈ 3.23964297272935
@test CovarianceMatrices.bwNeweyWest(lm1, ParzenKernel(), prewhite = false) ≈ 2.7987360579390486
@test CovarianceMatrices.bwNeweyWest(lm1, QuadraticSpectralKernel(), prewhite = false) ≈ 1.390324243706777

@test CovarianceMatrices.bwNeweyWest(lm1, BartlettKernel(), prewhite = true) ≈ 2.2830418148034246
@test CovarianceMatrices.bwNeweyWest(lm1, ParzenKernel(), prewhite = true) ≈ 3.390825323658861
@test CovarianceMatrices.bwNeweyWest(lm1, QuadraticSpectralKernel(), prewhite = true) ≈ 1.6844556099832346


V = vcov(lm1, VARHAC(1,1,1))
V = vcov(lm1, VARHAC(1,2,1))
V = vcov(lm1, VARHAC(1,3,1))

V = vcov(lm1, VARHAC(1,1,2))
V = vcov(lm1, VARHAC(1,2,2))
V = vcov(lm1, VARHAC(1,3,2))

V = vcov(lm1, VARHAC(1,1,3))
V = vcov(lm1, VARHAC(1,2,3))
V = vcov(lm1, VARHAC(1,3,3))


## X = CovarianceMatrices.ModelMatrix(lm1.model);
## u = CovarianceMatrices.wrkresidwts(lm1.model.rr);
## z = X.*u;

## vcov(lm1, TruncatedKernel(1.0), prewhite = false)
