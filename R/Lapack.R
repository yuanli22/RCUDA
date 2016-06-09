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

tGPU <- function(X)
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
