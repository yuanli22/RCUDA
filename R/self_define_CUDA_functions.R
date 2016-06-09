#' dividegpu
#'
#' This function computes the element-wise division of two given 
#' vectors/matrices by using self-defined CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise division of vectors/matrices (x / y),
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{multiplygpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' dividegpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

dividegpu <- function (x, y) 
{ 
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3]) 
      != as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "vector_divide", 
                x$ptr,
                y$ptr, 
                as.integer(x[2]) * as.integer(x[3]),
                PACKAGE = "supplement"
               )
  if (as.integer(x[3]) != 1){
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext<-GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' sumgpu
#'
#' Compute the summation of given vector/matrix  
#' 
#' This function computes the summation of given vector/matrix 
#' by using self-defined CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @return vector/matrix summation
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{meangpu}}  
#' @export
#' @examples
#' a <- creategpu(1:4)
#' sumgpu(a)

sumgpu <- function(x)
{
 checkGPU(x) 
 ext <- .Call(
                "vector_reduction",
                x$ptr,      
                as.integer(x[2]) * as.integer(x[3]),
                PACKAGE = "supplement" 
              )
 return(ext)
}


#' meangpu
#'
#' Compute the mean of given vector/matrix  
#' 
#' This function computes the mean of given vector/matrix 
#' by using self-defined CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @return vector/matrix mean
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{sumgpu}}  
#' @export
#' @examples
#' a <- creategpu(1:4)
#' meangpu(a)

meangpu <- function(x)
{
 checkGPU(x) 
 ext <- .Call(
                "vector_reduction",
                x$ptr,      
                as.integer(x[2]) * as.integer(x[3]),
                PACKAGE = "supplement" 
              )
 return(ext / (as.integer(x[2]) * as.integer(x[3])))
}


#' vargpu
#'
#' Compute the variance of given vector/matrix  
#' 
#' This function computes the variance of given vector/matrix 
#' by using self-defined CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @return vector/matrix variance
#' @author Yuan Li        
#' @keywords CUDA reduction 
#' @seealso \code{\link{sumgpu}}
#' @export
#' @examples
#' a <- creategpu(1:4)
#' vargpu(a)

vargpu <- function(x)
{
 checkGPU(x) 
 n <- as.integer(x[2]) * as.integer(x[3])
 mean <- sumgpu(x) / n
 ext <- .Call(
                "cudavarGPU",
                x$ptr, 
                n,
                as.numeric(mean),
                PACKAGE = "supplement" 
              )
 ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
 result <- sumgpu(ext) / (n-1) 
 return(result)
}


#' subsetgpu
#'
#' This function returns the specified subset of given GPU vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension
#' @param index index of the vector/matrix subset 
#' @return subset of the given vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{creategpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' subsetgpu(a_gpu,c(1, 2))->b_gpu
#' gathergpu(b_gpu)

subsetgpu <- function (input, index) 
{ 
    checkGPU(input)
    n <- length(index) 
    index <- as.integer(index)
    if ((as.integer(input[2]) * as.integer(input[3])
         < max(index)) | (min(index) < 1))
    	stop ("index out of bound")
    ext <- .Call(
                  "subset_GPU", 
                  input$ptr,
                  as.integer(n),
                  as.integer(index),
                  PACKAGE = "supplement"
                 )
    ext <- GPUobject(ext, as.integer(n), 1)
    return(ext)
  }
