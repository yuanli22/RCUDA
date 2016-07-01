#' geamgpu
#'
#' This function computes the matrix-matrix addition/trasportation 
#' C = a op ( A ) + b op ( B ) 
#' by using CUDA cublas function cublasDgeam
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension
#' @param C output matrix; list of R external GPU pointer and dimension
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{gemvgpu}}  
#' @export
#' @examples
#' A_gpu <- creategpu(1:6, 3, 2)
#' B_gpu <- creategpu(1:6, 3, 2)
#' C_gpu <- creategpu(1:4, 2, 2)
#' gemmgpu(2, 1, 1, A_gpu, B_gpu, beta=1, C_gpu)
#' gathergpu(C_gpu)

geamgpu <- function(transa = 1, transb = 1, alpha = 1, A, B, beta = 0, C)
{
  checkGPU(A)
  checkGPU(B)
  checkGPU(C)
  if ((transa != 1) && (transa != 2) && (transa != 3))
    stop ("A transpose operation input error")
  if ((transb != 1) && (transb != 2) && (transb != 3))
    stop ("B transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if ((transa == 1) && (transb == 1)) {
    if (!all.equal(as.integer(A[2]), as.integer(B[2]), 
        as.integer(C[2])))
      stop ("dimensions not match")
    if (!all.equal(as.integer(A[3]), as.integer(B[3]), 
        as.integer(C[3])))
      stop ("dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(A[3])
    }
  if ((transa != 1) && (transb == 1)) {
    if (!all.equal(as.integer(A[3]), as.integer(B[2]), 
        as.integer(C[2])))
      stop ("dimensions not match")
    if (!all.equal(as.integer(A[2]), as.integer(B[3]), 
        as.integer(C[3])))
      stop ("dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(A[2])
    }
  if ((transa == 1) && (transb != 1)) {
    if (!all.equal(as.integer(A[2]), as.integer(B[3]), 
        as.integer(C[2])))
      stop ("dimensions not match")
    if (!all.equal(as.integer(A[2]), as.integer(B[2]), 
        as.integer(C[3])))
      stop ("dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(A[3])
    }
  if ((transa != 1) && (transb != 1)) {
    if (!all.equal(as.integer(A[3]), as.integer(B[3]), 
        as.integer(C[2])))
      stop ("dimensions not match")
    if (!all.equal(as.integer(A[2]), as.integer(B[2]), 
        as.integer(C[3])))
      stop ("dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(A[2])
    }
  ext <- .Call(
                "geamGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(B[2]),
                 as.integer(C[2]),	
                 m,
                 n,	
                 as.numeric(transa),
                 as.numeric(transb),
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}




#' dgmmgpu
#'
#' This function performs the matrix-matrix multiplication 
#' C = A diag(x) or C = diag(x) A 
#' by using CUDA cublas function cublasDdgmm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param sidemode indicates whether the given matrix is on the left or right side
#' in the matrix equation solved by a particular function. If sidemode == 1, 
#' the matrix is on the left side in the equation If sidemode == 2, 
#' the matrix is on the right side in the equation.
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{symmgpu}}  
#' @export
#' @examples
#' A_gpu <- creategpu(1:9, 3, 3)
#' B_gpu <- creategpu(1:6, 3, 2)
#' C_gpu <- creategpu(1:4, 3, 2)
#' symmgpu(alpha=1, A_gpu, B_gpu, beta=1, C_gpu)
#' gathergpu(C_gpu)

dgmmgpu <- function(sidemode = 1, A, x, C)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(C)
  if (as.integer(x[2])!= 1)
    stop ("input x should be vector")
  if (!all.equal(as.integer(A[2]), as.integer(C[2])))
      stop ("A C dimensions not match")
  if (!all.equal(as.integer(A[3]), as.integer(C[3])))
      stop ("A C dimensions not match")
  if (sidemode == 1) {
    if (!identical(as.integer(A[3]), as.integer(x[2])))
      stop ("A x dimensions not match")
  }
  if (sidemode == 2) {
    if (!identical(as.integer(A[2]), as.integer(x[2])))
      stop ("A x dimensions not match")
  }
  ext <- .Call(
                "dgmmGPU",
                 A$ptr,
                 x$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(1),
                 as.integer(C[2]),			
                 as.integer(A[2]),
                 as.integer(A[3]),
                 as.numeric(sidemode)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' tgpu
#'
#' This function transposes the given matrix 
#' by using CUDA cublas cublasDgeam
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @return matrix transpose, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso  \code{\link{creategpu}} 
#' @export
#' @examples
#' a <- 1:12
#' a_gpu <- creategpu(a, 3, 4)
#' tgpu(a_gpu) -> c_gpu
#' gathergpu(c_gpu)

tgpu <- function(X)
{
  checkGPU(X)
  ext <- .Call(
                "tGPU",
                 X$ptr,
                 as.integer(X[2]),
                 as.integer(X[3])            
               )
   ext <- GPUobject(ext, as.integer(X[3]),as.integer(X[2]))
   return(ext)
}


#' inversegpu
#'
#' This function computes the inversion of given matrix (squared) 
#' by using CUDA cublas function cublasDgetrfBatched 
#' and cublasDgetriBatched (LU decomposition)
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @return matrix inversion, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{mmgpu}} \code{\link{creategpu}} 
#' @export
#' @examples
#' a <- 1:9
#' a_gpu <- creategpu(a, 3, 3)
#' inversegpu(a_gpu) -> c_gpu
#' gathergpu(c_gpu)

inversegpu<-function(X)
{
    checkGPU(X)
    if (as.integer(X[2]) != as.integer(X[3]))
    	stop ("only squared matrix is supported")
    ext <- .Call(
                  "inversGPU",
                  X$ptr,                         
                  as.integer(X[2])
                 )
    ext <- GPUobject(ext, as.integer(X[2]), as.integer(X[2]))
    return(ext)
}
