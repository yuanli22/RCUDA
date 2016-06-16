#' gemvgpu
#'
#' This function computes matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDgemv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gergpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 4, 3)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gemvgpu(A_gpu, x_gpu, y_gpu, 1, 1, 1)
#' gathergpu(y_gpu)

gemvgpu <- function(A, x, y, alpha = 1, beta = 0, trans = 1)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if ((trans != 1) && (trans != 2) && (trans != 3))
    stop ("transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if (trans == 1) {
    if (as.integer(A[3]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[2]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    } 
  if ((trans == 2) || (trans == 3)){
    if (as.integer(A[2]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[3]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    }
  ext <- .Call(
                "gemvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
                 as.integer(A[3]),
		   as.numeric(trans)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}


#' gbmvgpu
#'
#' This function computes banded matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDgbmv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param kl number of subdiagonals
#' @param ku number of superdiagonals
#' @param alpha scale factor a of banded matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gergpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 4, 3)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gemvgpu(A_gpu, x_gpu, y_gpu, 1, 1, 1)
#' gathergpu(y_gpu)

gbmvgpu <- function(A, x, y, kl, ku, alpha = 1, beta = 0, trans = 1)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if ((trans != 1) && (trans != 2) && (trans != 3))
    stop ("transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if (trans == 1) {
    if (as.integer(A[3]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[2]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    } 
  if ((trans == 2) || (trans == 3)){
    if (as.integer(A[2]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[3]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    }
  ext <- .Call(
                "gbmvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
                 as.integer(A[3]),
		   as.integer(kl),
	          as.integer(ku),
		   as.numeric(trans)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}


#' gergpu
#'
#' This function perform the the rank-1 update A = a x y T + A,
#' by using CUDA cublas function cublasDger
#' @param A input/output matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix A; default 1
#' @return updated matrix A, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix A's number of rows}
#' \item{n: }{matrix A's number of columns}
#' }
#' @seealso \code{\link{gemvgpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 3, 4)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gergpu(A_gpu, x_gpu, y_gpu, 1)
#' gathergpu(A_gpu)

gergpu <- function(A, x, y, alpha = 1)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if (as.integer(A[2]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  if (as.integer(A[3]) != as.integer(y[2]))
     stop ("A y dimension doesn't match")
  ext <- .Call(
                "gerGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.integer(A[2]),
                 as.integer(A[3])         
               )
   ext <- GPUobject(ext, as.integer(A[2]), as.integer(A[3]))
   return(ext)
}


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

