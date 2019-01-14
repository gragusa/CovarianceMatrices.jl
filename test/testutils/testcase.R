## Testing VCOV

library(sandwich)



build_vcov <- function(x, bw, prewhite) {
  
  if (bw == "auto" ) {
    bw_andrews <- function(x, kernel, prewhite) {
      bwAndrews(x, prewhite = prewhite, kernel = kernel)
    }
    bw_neweywest <- function(x, kernel, prewhite) {
      bwNeweyWest(x, prewhite = prewhite)
    }
  } else {
    if (is.numeric(bw)) {
      bw_andrews <- function(x, ...) bw
      bw_neweywest <- function(x, ...) bw
    }}
  
  andrews <- list()
  neweywest <- list()
  
  for (k in c("Truncated", "Parzen", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")) {
    tmp1 <- bw_andrews(x, kernel = k, prewhite = prewhite)
    andrews[[k]] = list(bw = tmp1, V = kernHAC(x, prewhite = prewhite, adjust = FALSE, kernel = k, bw = tmp1))
  }
  
  for (k in c("Parzen", "Bartlett", "Quadratic Spectral")) {
    tmp1 <- bw_neweywest(x, kernel = k, prewhite = prewhite)
    neweywest[[k]] = list(bw = tmp1, V = kernHAC(x, prewhite = prewhite, adjust = FALSE, kernel = k, bw = tmp1))
  }
  
  andrews$prewhite <- prewhite
  neweywest$prewhite <- prewhite
  andrews$bwtype <- ifelse(is.numeric(bw), "numeric", "auto")
  neweywest$bwtype <- ifelse(is.numeric(bw), "numeric", "auto")
  
  out <- list(andrews = andrews, neweywest = neweywest)
  out
}


build_lrcovariances <- function(x, bw, prewhite) {
  
  if (bw == "auto" ) {
    bw_andrews <- function(x, kernel, prewhite) {
      bwAndrews(lm(x~1), prewhite = prewhite, kernel = kernel)
    }
    bw_neweywest <- function(x, kernel, prewhite) {
      bwNeweyWest(lm(x~1), prewhite = prewhite)
    }
  } else {
    if (is.numeric(bw)) {
      bw_andrews <- function(x, ...) bw
      bw_neweywest <- function(x, ...) bw
    }}
  
  andrews <- list()
  neweywest <- list()

  for (k in c("Truncated", "Parzen", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")) {
    tmp1 <- bw_andrews(x, kernel = k, prewhite = prewhite)
    andrews[[k]] = list(bw = tmp1, V = lrvar(x, prewhite = prewhite, adjust = FALSE, kernel = k, bw = tmp1))
  }

  for (k in c("Parzen", "Bartlett", "Quadratic Spectral")) {
    tmp1 <- bw_neweywest(x, kernel = k, prewhite = prewhite)
    neweywest[[k]] = list(bw = tmp1, V = lrvar(x, prewhite = prewhite, adjust = FALSE, kernel = k, bw = tmp1))
  }
  
  andrews$prewhite <- prewhite
  neweywest$prewhite <- prewhite
  andrews$bwtype <- ifelse(is.numeric(bw), "numeric", "auto")
  neweywest$bwtype <- ifelse(is.numeric(bw), "numeric", "auto")
  
  out <- list(andrews = andrews, neweywest = neweywest)
  out
}



x <-
  c(
    0.164199142438412,
    -0.22231320001538,
    -2.29596418288347,
    -0.562717239408665,
    1.11832433510705,
    0.259810317798567,
    0.647885100553029,
    2.53438209392891,
    -0.419561292138475,
    1.19138801574376,
    2.52661839907567,
    -0.443382492040113,
    -0.137169008509379,
    -0.967782699857501,
    0.150507152028812,
    -1.27098181862663,
    1.38639734998711,
    0.231229342441316,
    -0.943827510026301,
    -1.11211457440442)

univariateout <- list(build_lrcovariances(x, bw = "auto", prewhite = 0),
                           build_lrcovariances(x, bw = "auto", prewhite = 1),
                           build_lrcovariances(x, bw = 1.5, prewhite = 0),
                           build_lrcovariances(x, bw = 1.5, prewhite = 1))



X <-
  structure(c(-0.392343481403554, 0.369702177262769, -0.283239954622452, 
              -1.71955159119934, -0.196311837779395, 0.567935573633728, 0.675556050609011, 
              -0.59266740997335, 0.433501752110958, -0.108134518170885, 0.686698254275096, 
              0.905166380758529, 0.997306068740448, -0.498167979332402, -0.52789448547848, 
              0.100420689892425, 1.49495246674266, -0.601579368927597, -0.166121087348934, 
              0.545348413754815, 0.294595939676804, -1.4327640501377, 0.719755536433108, 
              -1.2570984424727, -1.5357895599007, 1.0027040264714, 1.08593183743541, 
              -0.711342086638968, -0.772154611498373, 1.30668004120162, 2.89166460485415, 
              0.614941151434828, -1.59157215625112, -0.51709522643053, 1.97543651877893, 
              1.92940239570577, 0.889679654045449, -0.471409046820902, -1.30435924644088, 
              0.424385975482526, -1.63421078193559, -0.562676024339334, -1.9773880104471, 
              -1.13903619779294, 0.586555209142932, -1.60335681344097, -1.19189517293108, 
              2.13456541115109, -1.42078068631219, -0.207019328929601, -0.600736667890819, 
              -1.41872438508684, -0.608094864262906, 1.07318477908557, -0.477244503174433, 
              0.567645883530209, -0.149728929145769, 1.41918460373266, 0.462399751718563, 
              -0.132320093478005, 1.27839375300926, -0.480093346919616, -0.0428876036353262, 
              -1.56004471316687, -0.134394994431227, 2.04942053641657, -1.8022396128532, 
              -1.72537103051051, -0.657108341488561, 1.34392729540088, 1.90019159830845, 
              0.126395933995686, -0.770308826277006, 0.457784471260252, -0.160271362465108, 
              0.0598471594810112, 1.75048422563306, -0.737657566749125, -0.462941254536989, 
              0.699940101308202, 1.2689352813542, -0.336296482224711, 1.42781944149594, 
              -1.69159952993731, -1.15816200645816, 0.83309270822555, -1.34670872662577, 
              2.24540522326547, 1.14409536415596, -0.959417691691381), .Dim = c(30L, 3L))


multivariateout <- list(build_lrcovariances(X, bw = "auto", prewhite = 0),
                       build_lrcovariances(X, bw = "auto", prewhite = 1),
                         build_lrcovariances(X, bw = 1.5, prewhite = 0),
                         build_lrcovariances(X, bw = 1.5, prewhite = 1))

w <-
  c(0.878586419392377, 0.16248927381821, 0.0735696377232671, 0.588177684927359, 
    0.807603721972555, 0.790315046673641, 0.0271224896423519, 0.179313898319378, 
    0.642000824445859, 0.956059637246653, 0.429024756653234, 0.336806409992278, 
    0.285446277353913, 0.296879382571205, 0.845753491157666, 0.768537326948717, 
    0.381573883583769, 0.44569466332905, 0.896755887661129, 0.421602969290689, 
    0.94799219397828, 0.127784259151667, 0.84408991644159, 0.186006092932075, 
    0.533113194862381, 0.585947344079614, 0.0927620485890657, 0.745519452961162, 
    0.481594698270783, 0.0099785930942744)


Y <-
  c(1.14063384846019, -1.98711855056335, 0.256690117099067, 0.333148570730093, 
    1.17066829261603, -0.741305574283845, 0.201119741293501, 1.14674492697252, 
    -3.10438570397866, 1.06728960677537, -0.789078543391792, 0.495519546755053, 
    -1.10427745778408, -1.79345103551701, 1.96216508427282, 0.7910731550863, 
    -0.0323701138692571, 1.23591606066751, 1.44724920199179, -1.18675936518181, 
    -1.1091245417785, 0.208367234262105, 0.152690630525284, 1.40240050394334, 
    1.16394141583204, -0.992333328196718, 1.21330671046369, -1.22531524573956, 
    -1.71491353921502, 0.23890444712464)

df <- data_frame(y = Y, x1 = X[,1], x2 = X[,2], x3 = X[,3], w = w)


regout <- list(build_vcov(lm(y~x1+x2+x3, data = df), bw = "auto", prewhite = 0),
               build_vcov(lm(y~x1+x2+x3, data = df), bw = "auto", prewhite = 1),
               build_vcov(lm(y~x1+x2+x3, data = df), bw = 1.5, prewhite = 0),
               build_vcov(lm(y~x1+x2+x3, data = df), bw = 1.5, prewhite = 1))


wregout <- list(build_vcov(lm(y~x1+x2+x3, weights = w, data = df), bw = "auto", prewhite = 0),
               build_vcov(lm(y~x1+x2+x3, weights = w, data = df), bw = "auto", prewhite = 1),
               build_vcov(lm(y~x1+x2+x3, weights = w, data = df), bw = 1.5, prewhite = 0),
               build_vcov(lm(y~x1+x2+x3, weights = w, data = df), bw = 1.5, prewhite = 1))


write(x = jsonlite::toJSON(univariateout, digits = 30), file = "univariate.json")
write(jsonlite::toJSON(multivariateout, digits = 30), file = "multivariate.json")
write(jsonlite::toJSON(regout, digits = 30), file = "regression.json")
write(jsonlite::toJSON(wregout, digits = 30), file = "wregression.json")

write_csv(df, path = "~/.julia/dev/CovarianceMatrices/test/testdata/ols_df.csv")
