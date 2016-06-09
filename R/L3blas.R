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