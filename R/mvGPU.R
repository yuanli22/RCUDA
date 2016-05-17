
#' mvGPU
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
#' @seealso \code{\link{mmGPU}} \code{\link{createGPUmat}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:3
#' a_gpu <- createGPUmat(a,2,2)
#' b_gpu <- createGPU(b)
#' mvGPU(a_gpu, b_gpu)->c_gpu
#' gatherGPU(c_gpu)



mvGPU <- function(X, y)
{
  checkGPU(X)
  checkGPU(y)
  if (as.integer(X[3])!=as.integer(y[2]))
  stop ("dimension doesn't match")
  ext <- .Call(
              "mvGPU",
               X$ptr,
               y$ptr,
               as.integer(X[2]),
               as.integer(X[3])            
            )
   ext<-GPUobject(ext, as.integer(X[2]),1)
   return(ext)
}
