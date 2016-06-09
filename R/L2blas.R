#' addgpu
#'
#' This function computes the element-wise addition of two given 
#' vectors/matrices by using CUDA cublas function cublasDgeam
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise addition of two vectors/matrices (x + y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{subtractgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' addgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

addgpu <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      != as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "addGPU",
                x$ptr,
                y$ptr,             
                as.integer(x[2]),
                as.integer(x[3])
               )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' subtractgpu
#'
#' This function computes the element-wise subtraction of two
#' given vectors/matrices by using CUDA cublas function cublasDgeam
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise subtraction of vectors or matrices (x - y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{addgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' subtractgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

subtractgpu<-function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      !=as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "subtractGPU",
                x$ptr,
                y$ptr,             
                as.integer(x[2]),
                as.integer(x[3])
              )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' multiplygpu
#'
#' This function computes the element-wise multiplication of 
#' two given vectors/matricesby using CUDA cublas function cublasDdgmm
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise multiplication of vectors/matrices (x * y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{dividegpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' multiplygpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

multiplygpu <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      != as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "vvGPU",
                 x$ptr,
                 y$ptr,
                 as.integer(x[2]) * as.integer(x[3])
              )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' mvgpu
#'
#' This function computes the matrix-vector multiplication (X * y) 
#' by using CUDA cublas function cublasDgemv
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @param y input vector; list of R external GPU pointer and dimension
#' @return matrix-vector multiplication (X * y), a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns; vector y's number of elements}
#' }
#' @seealso \code{\link{mmgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:3
#' a_gpu <- creategpu(a, 2, 2)
#' b_gpu <- creategpu(b)
#' mvgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

mvgpu <- function(X, y)
{
  checkGPU(X)
  checkGPU(y)
  if (as.integer(X[3]) != as.integer(y[2]))
    stop ("dimension doesn't match")
  ext <- .Call(
                "mvGPU",
                 X$ptr,
                 y$ptr,
                 as.integer(X[2]),
                 as.integer(X[3])            
               )
   ext <- GPUobject(ext, as.integer(X[2]), 1)
   return(ext)
}

