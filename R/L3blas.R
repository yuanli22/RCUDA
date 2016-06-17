#' gemmgpu
#'
#' This function computes the matrix-matrix multiplication 
#' C = a op ( A ) op ( B ) + b C 
#' by using CUDA cublas function cublasDgemm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix Y's number of columns}
#' }
#' @seealso \code{\link{mmgpu}}  
#' @export
#' @examples
#' a <- 1:6
#' b <- 2:7
#' a_gpu <- creategpu(a, 2, 3)
#' b_gpu <- creategpu(b, 3, 2)
#' mmgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

gemmgpu <- function(A, B, C, transa = 1, transb = 1, alpha = 1, beta = 0)
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
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[3]) != as.integer(B[2]))
      stop ("A and B dimensions not match")
    if (as.integer(B[3]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(B[3])
    k <- as.integer(A[3])
    }
  if ((transa == !1) && (transb == 1)) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[2]) != as.integer(B[2]))
      stop ("A and B dimensions not match")
    if (as.integer(B[3]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(B[3])
    k <- as.integer(A[2])
    }
  if ((transa == 1) && (transb == !1)) {
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[3]) != as.integer(B[3]))
      stop ("A and B dimensions not match")
    if (as.integer(B[2]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(B[2])
    k <- as.integer(A[3])
    }
  if ((transa == !1) && (transb == !1)) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[2]) != as.integer(B[3]))
      stop ("A and B dimensions not match")
    if (as.integer(B[2]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(B[2])
    k <- as.integer(A[2])
    }
  ext <- .Call(
                "gemmGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 m,
                 n,
                 k,
                 as.numeric(transa),
                 as.numeric(transb),
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}

#' mmgpu
#'
#' This function computes the matrix-matrix multiplication (X * Y) 
#' by using CUDA cublas function cublasDgemm
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @param Y input matrix; list of R external GPU pointer and dimension
#' @return matrix-matrix multiplication (X * Y), a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix Y's number of columns}
#' }
#' @seealso \code{\link{mmgpu}}  
#' @export
#' @examples
#' a <- 1:6
#' b <- 2:7
#' a_gpu <- creategpu(a, 2, 3)
#' b_gpu <- creategpu(b, 3, 2)
#' mmgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

mmgpu <- function(X, Y)
{
  checkGPU(X)
  checkGPU(Y)
  if (as.integer(X[3]) != as.integer(Y[2]))
    stop ("dimension doesn't match")
  ext <- .Call(
                "mmGPU",
                 X$ptr,
                 Y$ptr,
                 as.integer(X[2]),
                 as.integer(Y[3]),
                 as.integer(X[3])            
              )
   ext <- GPUobject(ext, as.integer(X[2]), as.integer(Y[3]))
   return(ext)
}