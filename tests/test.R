rm(list = ls())
install.packages("/mnt/home/yli16/yli16/Rcublas_0.1.tar.gz", repos = NULL,INSTALL_opts = c('--no-lock'))
library(Rcublas)
gpuQuery()
 
## define some constant
eps <- 1e-5

set.seed(1424)
##define test function to automatically print out test result
GPUtest<-function(a, b, c)
{
  if(a)
  {cat(b, "\n")}
   else
  {cat(c, "\n")}
}
##define multiple run function
multiplerun<-function(a, n)
{
for (i in 1:n)
   {
       a
    }
}

L1testout <- matrix("NA", 40, 5)
colnames(L1testout) <- c("Function", "Length", "Speedup","timeCPU","timeGPU")
runtime <- 40
vec <- c(1e+5,5e+5,1e+6,3e+6,5e+6,1e+7,5e+7,1e+8)/1e+3
for (k in 1:length(vec))
{
##test GPU create and gather function
a <- rnorm(vec[k])
creategpu(a)->x
gathergpu(x) -> a_gpu
((sum(a==a_gpu))==length(a))->result
GPUtest(result, "create and gather function run perfectly!\n",
        "create and gather function fail!\n")



##test level 1 cublas functions
##test min 
(abs(min(abs(a)) - abs(a[minGPU(x)])) < eps) -> result
GPUtest(result, "min function run perfectly!",
        "min function fail!")
system.time(multiplerun(min(abs(a)), runtime)) -> timecpu
system.time(multiplerun(minGPU(x), runtime)) -> timegpu
cat("min function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L1testout[k,1] <- c("Minimum")
L1testout[k,2] <- vec[k]
L1testout[k,3] <- timecpu[3] / timegpu[3]
L1testout[k,4] <- timecpu[3] 
L1testout[k,5] <- timegpu[3]
 

##test max
(abs(max(abs(a))-abs(a[maxGPU(x)]))<eps)->result
GPUtest(result,"max function run perfectly!",
        "max function fail!")
system.time(multiplerun(max(abs(a)), runtime)) -> timecpu
system.time(multiplerun(maxGPU(x), runtime)) -> timegpu
cat("max function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L1testout[length(vec)+k,1] <- c("Maximum")
L1testout[length(vec)+k,2] <- vec[k]
L1testout[length(vec)+k,3] <- timecpu[3] / timegpu[3]
L1testout[length(vec)+k,4] <- timecpu[3]  
L1testout[length(vec)+k,5] <- timegpu[3]
 


##test norm2
(abs(sqrt(sum(a * a))-norm2GPU(x)) < eps) -> result
GPUtest(result,"norm2 function run perfectly!",
        "norm2 function fail!")
system.time(multiplerun(sqrt(sum(a * a)), runtime)) -> timecpu
system.time(multiplerun(norm2GPU(x), runtime)) -> timegpu
cat("norm2 function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L1testout[2*length(vec)+k,1] <- c("norm2")
L1testout[2*length(vec)+k,2] <- vec[k]
L1testout[2*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L1testout[2*length(vec)+k,4] <- timecpu[3]  
L1testout[2*length(vec)+k,5] <- timegpu[3]
 



##test dot
(abs(sum(a * a) - dotGPU(x, x)) < eps) -> result
GPUtest(result,"dot function run perfectly!",
        "dot function fail!")
system.time(multiplerun(sum(a * a), runtime)) -> timecpu
system.time(multiplerun(dotGPU(x, x), runtime)) -> timegpu
cat("dot function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L1testout[3*length(vec)+k,1] <- c("dot")
L1testout[3*length(vec)+k,2] <- vec[k]
L1testout[3*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L1testout[3*length(vec)+k,4] <- timecpu[3]  
L1testout[3*length(vec)+k,5] <- timegpu[3]




##test vector scale
alpha<-rnorm(1)
(abs(sum(a * alpha - gathergpu(scaleGPU(x , alpha)))) / vec[k] < eps) -> result
GPUtest(result,"scale function run perfectly!",
        "scale function fail!")
system.time(multiplerun(a * alpha, runtime)) -> timecpu
system.time(multiplerun(scaleGPU(x, alpha), runtime)) -> timegpu
cat("scale function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L1testout[4*length(vec)+k,1] <- c("scale")
L1testout[4*length(vec)+k,2] <- vec[k]
L1testout[4*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L1testout[4*length(vec)+k,4] <- timecpu[3] 
L1testout[4*length(vec)+k,5] <- timegpu[3]
}




##test level 2 cublas functions

runtime <- 1
vec <- c(5e+5,7e+5,1e+6,3e+6,5e+6,7e+6,1e+7,3e+7)/1e+3
L2testout <- matrix("NA", 40, 5)
colnames(L2testout) <- c("Function", "Length", "Speedup","timeCPU","timeGPU")
for (k in 1:length(vec))
{

##test addition
a <- rnorm(vec[k])
b <- rnorm(vec[k])
creategpu(a) -> x
creategpu(b) -> y
addGPU(x, y) -> z
(sum(abs(gathergpu(z ) - (a + b)) < eps) == vec[k]) -> result
GPUtest(result,"vector addition function run perfectly!",
        "vector addition function fail!")
system.time(multiplerun(a + b, runtime) )-> timecpu
system.time(multiplerun(addGPU(x, y ), runtime)) -> timegpu
cat("vector addition function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L2testout[0*length(vec)+k,1] <- c("vector addition")
L2testout[0*length(vec)+k,2] <- vec[k]
L2testout[0*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L2testout[0*length(vec)+k,4] <- timecpu[3]  
L2testout[0*length(vec)+k,5] <- timegpu[3] 



##test vector subtraction
subtractGPU(x, y) -> z
(sum(abs(gathergpu(z ) - (a - b)) < eps) == vec[k]) -> result
GPUtest(result,"vector subtraction function run perfectly!", 
"vector subtraction function fail!")
system.time(multiplerun((a - b), runtime)) -> timecpu
system.time(multiplerun((subtractGPU(x, y )), runtime)) -> timegpu
cat("vector subtraction function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L2testout[1*length(vec)+k,1] <- c("vector subtraction")
L2testout[1*length(vec)+k,2] <- vec[k]
L2testout[1*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L2testout[1*length(vec)+k,4] <- timecpu[3]  
L2testout[1*length(vec)+k,5] <- timegpu[3]
 


##test vector multiplication 
multiplyGPU(x, y) -> z
(sum(abs(gathergpu(z ) - (a * b)) < eps) == vec[k]) -> result
GPUtest(result,"vector multiplication function run perfectly!",
        "vector multiplication function fail!")
system.time(multiplerun((a * b), runtime*10000)) -> timecpu
system.time(multiplerun((multiplyGPU(x, y )), runtime*10000)) -> timegpu
cat("vector multiplication function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L2testout[2*length(vec)+k,1] <- c("vector multiplication")
L2testout[2*length(vec)+k,2] <- vec[k]
L2testout[2*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L2testout[2*length(vec)+k,4] <- timecpu[3] 
L2testout[2*length(vec)+k,5] <- timegpu[3]


##test vector division
divideGPU(x, y) -> z
(sum(abs(gathergpu(z) - (a / b)) < eps) == vec[k]) -> result
GPUtest(result,"vector division function run perfectly!",
        "vector division function fail!")
system.time(multiplerun((a / b), runtime)) -> timecpu
system.time(multiplerun((divideGPU(x, y)), runtime)) -> timegpu
cat("vector division function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L2testout[3*length(vec)+k,1] <- c("vector division")
L2testout[3*length(vec)+k,2] <- vec[k]
L2testout[3*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L2testout[3*length(vec)+k,4] <- timecpu[3]  
L2testout[3*length(vec)+k,5] <- timegpu[3]
 }

##test matrix * vector  
runtime <- 500
vec <- c(2e+3,3e+3,5e+3,7e+3,8e+3,9e+3,1e+4,2e+4) /1e+2
for (k in 1:length(vec))
{
m <- vec[k]
n <- vec[k]
j <- vec[k]
a <- rnorm(n)
b <- rnorm(m)
matrixA <- matrix(rnorm(m * n), m, n)
matrixB <- matrix(rnorm(n * j), n, j)
creategpu(as.vector(matrixA), m, n) -> x
creategpu(a) -> y
mvGPU(x, y) -> z
gathergpu(z) -> c
(sum(abs(matrixA %*% a - c) < eps) == m) -> result
GPUtest(result,"M * V function run perfectly!",
        "M * V function fail!")
system.time(multiplerun(matrixA %*% a, runtime)) -> timecpu
system.time(multiplerun(mvGPU(x, y), runtime)) -> timegpu
cat("M * V function for matrix dimension", m,"*", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L2testout[4*length(vec)+k,1] <- c("matrix * vector")
L2testout[4*length(vec)+k,2] <- vec[k]
L2testout[4*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L2testout[4*length(vec)+k,4] <- timecpu[3]  
L2testout[4*length(vec)+k,5] <- timegpu[3]
}



L3testout <- matrix("NA", 15, 5)
colnames(L3testout) <- c("Function", "Length", "Speedup","timeCPU","timeGPU")
##test level 3 cublas functions
##test M * M 
runtime <- 1
vec <- c(3e+2,5e+2,1e+3,2e+3,3e+3)/1e+1
for (k in 1:length(vec))
{
m <- vec[k]
n <- vec[k]
j <- vec[k]

matrixA <- matrix(runif(m * n), m, n)
matrixB <- matrix(runif(n * j), n, j)
creategpu(as.vector((matrixA)), m, n) -> x
creategpu(as.vector((matrixB)), n, j) -> y
mmGPU(x, y) -> z
gathergpu(z) -> c
(sum(abs(as.vector(matrixA %*% matrixB) - c) < eps) == m * j) -> result
GPUtest(result,"M * M function run perfectly!",
        "M * M function fail!")
system.time(multiplerun(matrixA %*% matrixB, runtime)) -> timecpu
system.time(multiplerun(mmGPU(x, y ), runtime)) -> timegpu
cat("M * M function for matrix dimension", m,"*", n, "*", j,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L3testout[0*length(vec)+k,1] <- c("matrix * matrix")
L3testout[0*length(vec)+k,2] <- vec[k]
L3testout[0*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L3testout[0*length(vec)+k,4] <- timecpu[3] 
L3testout[0*length(vec)+k,5] <- timegpu[3]
}



##test transpose
runtime <- 1
vec <- c(1e+3,2e+3,3e+3,5e+3,7e+3)/1e+2
for (k in 1:length(vec))
{
m <- vec[k]
n <- vec[k]
matrixA <- matrix(runif(m * n), m, n)
creategpu(as.vector((matrixA)), m, n) -> x
tGPU(x)->z
gathergpu(z)->tx
(sum(abs(as.vector(t(matrixA)) - tx)) < eps) -> result
GPUtest(result,"transpose function run perfectly!",
        "transpose function fail!")
system.time(multiplerun(t(matrixA), runtime)) -> timecpu
system.time(multiplerun(tGPU(x), runtime)) -> timegpu
cat("transpose function for for matrix dimension", m,"*", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L3testout[1*length(vec)+k,1] <- c("matrix transpose")
L3testout[1*length(vec)+k,2] <- vec[k]
L3testout[1*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L3testout[1*length(vec)+k,4] <- timecpu[3]  
L3testout[1*length(vec)+k,5] <- timegpu[3]


##test cublas extension functions
##test inverse of matrix

rnorm(n * n, 500,141) -> a
creategpu(a,n,n) -> x
matrix(a, n, n) -> a
inverseGPU(x) -> y
gathergpu(y) -> b
(sum(abs(as.vector(solve(a)) - as.vector(b))) < eps) -> result
GPUtest(result,"inverse function run perfectly!",
        "inverse function fail!")
system.time(multiplerun(solve(a), runtime)) -> timecpu
system.time(multiplerun(inverseGPU(x), runtime)) -> timegpu
cat("inverse function for for matrix dimension", n,"*", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")

L3testout[2*length(vec)+k,1] <- c("matrix inverse")
L3testout[2*length(vec)+k,2] <- vec[k]
L3testout[2*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L3testout[2*length(vec)+k,4] <- timecpu[3] 
L3testout[2*length(vec)+k,5] <- timegpu[3]
}





L4testout <- matrix("NA", 20, 5)
colnames(L4testout) <- c("Function", "Length", "Speedup","timeCPU","timeGPU")
##compare performance of random generator
runtime <- 50
vec <- c(1e+3,1e+4,1e+5,2e+6,1e+7)/1e+3
for (k in 1:length(vec))
{
n <- vec[k]
system.time(multiplerun(rnorm(n), runtime)) -> timecpu
system.time(multiplerun(normRNGGPU(n, 1, 0, 1), runtime)) -> timegpu
cat("generating", n, "normal number",
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L4testout[0*length(vec)+k,1] <- c("normal random numbers generator")
L4testout[0*length(vec)+k,2] <- vec[k]
L4testout[0*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L4testout[0*length(vec)+k,4] <- timecpu[3] 
L4testout[0*length(vec)+k,5] <- timegpu[3]
 

##compare log norm
system.time(multiplerun(rlnorm(n), runtime)) -> timecpu
system.time(multiplerun(lognormRNGGPU(n, 1, 0, 1), runtime)) -> timegpu
cat("generating", n, "lognormal number",
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L4testout[1*length(vec)+k,1] <- c("log-normal random numbers generator")
L4testout[1*length(vec)+k,2] <- vec[k]
L4testout[1*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L4testout[1*length(vec)+k,4] <- timecpu[3] 
L4testout[1*length(vec)+k,5] <- timegpu[3]
 

##compare uniform
system.time(multiplerun(runif(n), runtime)) -> timecpu
system.time(multiplerun(uniformRNGGPU(n, 1), runtime)) -> timegpu
cat("generating", n, "uniform number",
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L4testout[2*length(vec)+k,1] <- c("uniform random numbers generator")
L4testout[2*length(vec)+k,2] <- vec[k]
L4testout[2*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L4testout[2*length(vec)+k,4] <- timecpu[3]  
L4testout[2*length(vec)+k,5] <- timegpu[3]
 

##compare poisson
system.time(multiplerun(rpois(n, 1), runtime)) -> timecpu
system.time(multiplerun(poissonRNGGPU(n, 1, 1), runtime)) -> timegpu
cat("generating", n, "poisson number",
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L4testout[3*length(vec)+k,1] <- c("Poisson random numbers generator")
L4testout[3*length(vec)+k,2] <- vec[k]
L4testout[3*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L4testout[3*length(vec)+k,4] <- timecpu[3]  
L4testout[3*length(vec)+k,5] <- timegpu[3]
}



L5testout <- matrix("NA", 25, 5)
colnames(L5testout) <- c("Function", "Length", "Speedup","timeCPU","timeGPU")
##test CUDA supplemental functions
##test element-wise exponential 
runtime <- 100
vec <- c(5e+4,1e+5,1e+6,1e+7,3e+7)/1e+2
for (k in 1:length(vec))
{
n <- vec[k]
abs(rnorm(n, 1, 1)) -> a
ag <- creategpu(a)
expGPU(ag)->result
gathergpu(result)->resultgpu
(sum(abs(resultgpu - exp(a))) < eps) -> result
GPUtest(result,"element-wise exponential function run perfectly!",
        "element-wise exponential fail!")
system.time(multiplerun(exp(a), runtime)) -> timecpu
system.time(multiplerun(expGPU(ag), runtime)) -> timegpu
cat("element-wise exponential function for vector of size", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L5testout[0*length(vec)+k,1] <- c("element-wise exponential")
L5testout[0*length(vec)+k,2] <- vec[k]
L5testout[0*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L5testout[0*length(vec)+k,4] <- timecpu[3] 
L5testout[0*length(vec)+k,5] <- timegpu[3]
 


##test element-wise log 
logGPU(ag)->result
gathergpu(result)->resultgpu
(sum(abs(resultgpu - log(a))) < eps) -> result
GPUtest(result,"element-wise log function run perfectly!",
        "element-wise log fail!")
system.time(multiplerun(log(a), runtime)) -> timecpu
system.time(multiplerun(logGPU(ag), runtime)) -> timegpu
cat("element-wise log function for vector of size", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L5testout[1*length(vec)+k,1] <- c("element-wise log")
L5testout[1*length(vec)+k,2] <- vec[k]
L5testout[1*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L5testout[1*length(vec)+k,4] <- timecpu[3] 
L5testout[1*length(vec)+k,5] <- timegpu[3]


##test element-wise sqrt 
sqrtGPU(ag)->result
gathergpu(result)->resultgpu
(sum(abs(resultgpu - sqrt(a))) < eps) -> result
GPUtest(result,"element-wise sqrt function run perfectly!",
        "element-wise sqrt fail!")
system.time(multiplerun(sqrt(a), runtime)) -> timecpu
system.time(multiplerun(sqrtGPU(ag), runtime)) -> timegpu
cat("element-wise sqrt function vector of size", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L5testout[2*length(vec)+k,1] <- c("element-wise sqrt")
L5testout[2*length(vec)+k,2] <- vec[k]
L5testout[2*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L5testout[2*length(vec)+k,4] <- timecpu[3] 
L5testout[2*length(vec)+k,5] <- timegpu[3]



##test element-wise power 
alpha <- runif(1)
powerGPU(ag, alpha)->result
gathergpu(result)->resultgpu
(sum(abs(resultgpu - a^alpha)) < eps) -> result
GPUtest(result,"element-wise power function run perfectly!",
        "element-wise power fail!")
system.time(multiplerun(a^alpha, runtime)) -> timecpu
system.time(multiplerun(powerGPU(ag, alpha), runtime)) -> timegpu
cat("element-wise power function vector of size", n,
    "speed up is(CPU time / GPU time)", timecpu[3] / timegpu[3], "\n")
L5testout[3*length(vec)+k,1] <- c("element-wise power")
L5testout[3*length(vec)+k,2] <- vec[k]
L5testout[3*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L5testout[3*length(vec)+k,4] <- timecpu[3] 
L5testout[3*length(vec)+k,5] <- timegpu[3]
}

##test vector reduction sum
runtime <- 100
vec <- c(5e+5,1e+6,5e+6,1e+7,5e+7)/1e+2
for (k in 1:length(vec))
{
n <- vec[k]
abs(rnorm(n, 1, 1)) -> a
ag <- creategpu(a)
(abs(sum(a)-sumGPU(ag)) < eps) -> result
GPUtest(result,"reduction sum function run perfectly!",
        "reduction sum function fail!")
system.time(multiplerun(sum(a), runtime)) -> timecpu
system.time(multiplerun(sumGPU(ag), runtime)) -> timegpu
cat("reduction sum function for vector of length", vec[k],
    "speed up is(CPU time / GPU time)",timecpu[3] / timegpu[3], "\n")
L5testout[4*length(vec)+k,1] <- c("vector reduction sum")
L5testout[4*length(vec)+k,2] <- vec[k]
L5testout[4*length(vec)+k,3] <- timecpu[3] / timegpu[3]
L5testout[4*length(vec)+k,4] <- timecpu[3] 
L5testout[4*length(vec)+k,5] <- timegpu[3]
}

 
