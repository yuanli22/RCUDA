#' rnormgpu
#'
#' This function generates normally distributed random numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT 
#' and curandGenerateNormalDouble
#' @param n number of random numbers 
#' @param mean mean of normal distribution; default value 0 
#' @param sd standard deviation of normal distribution; default value 1 
#' @param seed random number generator seed; default value 1 
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{rlognormgpu}}  
#' @export
#' @examples
#' a_gpu <- rnormgpu(100, 0, 1, 15)
#' gathergpu(a_gpu)

rnormgpu <- function(n, mean = 0, sd = 1, seed = 1)
{
    ext <- .Call(
                  "normRNGGPU",                        
                  as.integer(n),
	           as.numeric(mean),
	           as.numeric(sd),
	           as.numeric(seed)
                )
    ext<-GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}


#' rlognormgpu
#'
#' This function generates log-normally distributed random numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT 
#' and curandGenerateLogNormalDouble
#' @param n number of random numbers 
#' @param mean mean of log-normal distribution; default value 0 
#' @param sd standard deviation of log-normal distribution; default value 1 
#' @param seed random number generator seed; default value 1
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{rnormgpu}} 
#' @export
#' @examples
#' a_gpu <- rlognormgpu(100, 0, 1, 15)
#' gathergpu(a_gpu)

rlognormgpu <- function(n, mean = 0, sd = 1, seed = 1)
{

    ext <- .Call(
                  "lognormRNGGPU",                        
                  as.integer(n),
	           as.numeric(mean),
	           as.numeric(sd),
	           as.numeric(seed)
                )
    ext <- GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}


#' runifgpu
#'
#' This function generates uniformly distributed random numbers between 0 and 1 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT 
#' and curandGenerateUniformDouble
#' @param n number of random numbers 
#' @param seed random number generator seed; default value 1
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{creategpu}}  
#' @export
#' @examples
#' a_gpu <- runifgpu(100, 15)
#' gathergpu(a_gpu)

runifgpu <- function(n, seed = 1)
{
    ext <- .Call(
                  "uniformRNGGPU",                        
                  as.integer(n),
                  as.numeric(seed)
                )
    ext <- GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}


#' rpoisgpu
#'
#' This function generates Poisson distributed random numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT 
#' and curandGeneratePoisson
#' @param n number of random numbers 
#' @param lambda mean of Poisson distribution; default value 1 
#' @param seed random number generator seed; default value 1
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{runifgpu}} 
#' @export
#' @examples
#' a_gpu <- rpoisgpu(100, 1) 

rpoisgpu <- function(n, lambda = 1, seed = 1)
{
    ext <- .Call(
                  "poissonRNGGPU",                        
                  as.integer(n),
	           as.numeric(lambda),
	           as.numeric(seed)
                )
    ext <- GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}


#' rgammagpu
#'
#' This function generates Gamma distributed random numbers 
#' by using self-defined CUDA function based on George Marsaglia 
#' and Wai Wan Tsang's method 
#' @param n number of random numbers 
#' @param k shape parameter of Gamma distribution; default value 1
#' @param theta scale parameter of Gamma distribution; default value 1  
#' @param seed random number generator seed; default value 1
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{runifgpu}} 
#' @export
#' @examples
#' a_gpu <- rgammagpu(100, 2, 1) 

rgammagpu <- function(n, k = 1, theta = 1, seed = 1)
{
    ext <- .Call(
                  "gammaRNGGPU",                        
                  as.integer(n),
	           as.numeric(k),
	           as.numeric(theta),
	           as.numeric(seed),
                  PACKAGE = "supplement"
                )
    ext <- GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}



#' rbetagpu
#'
#' This function generates Beta distributed random numbers 
#' by using self-defined CUDA function based on George Marsaglia 
#' and Wai Wan Tsang's method and gamma/beta relationship 
#' @param n number of random numbers 
#' @param alpha shape parameter of Gamma distribution; default value 1
#' @param beta shape parameter of Gamma distribution; default value 1  
#' @param seed random number generator seed; default value 1
#' @return generated random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{runifgpu}} 
#' @export
#' @examples
#' a_gpu <- rbetagpu(100, 2, 1) 

rbetagpu <- function(n, alpha = 1, beta = 1, seed = 1)
{
    ext <- .Call(
                  "betaRNGGPU",                        
                  as.integer(n),
	           as.numeric(alpha),
	           as.numeric(beta),
	           as.numeric(seed),
                  PACKAGE = "supplement"
                )
    ext <- GPUobject(ext, as.integer(n), as.integer(1))
    return(ext)
}





