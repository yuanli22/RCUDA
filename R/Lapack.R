
#' tGPU
#'
#' This function transposes the given matrix 
#' by using CUDA cublas cublasDgeam
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @return matrix transpose, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns}
#' }
#' @seealso  \code{\link{createGPUmat}} 
#' @export
#' @examples
#' a <- 1:12
#' a_gpu <- createGPUmat(a,3,4)
#' tGPU(a_gpu)->c_gpu
#' gatherGPU(c_gpu)


tGPU <- function(X)
{
  checkGPU(X)
  ext <- .Call(
              "tGPU",
               X$ptr,
               as.integer(X[2]),
               as.integer(X[3])            
            )
   ext<-GPUobject(ext, as.integer(X[3]),as.integer(X[2]))
   return(ext)
}



#' inverseGPU
#'
#' This function computes the inverse of given matrix (square) 
#' by using CUDA cublas function cublasDgetrfBatched and cublasDgetriBatched (LU decomposition)
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @return matrix inverse, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns}
#' }
#' @seealso \code{\link{mmGPU}} \code{\link{createGPUmat}} 
#' @export
#' @examples
#' a <- 1:9
#' a_gpu <- createGPUmat(a,3,3)
#' inverseGPU(a_gpu)->c_gpu
#' gatherGPU(c_gpu)



inverseGPU<-function(X)
{
    checkGPU(X)
    if (as.integer(X[2])!=as.integer(X[3]))
    stop ("only square matrix supported")
    ext<-.Call(
                "inversGPU",
                X$ptr,                         
                as.integer(X[2])
              )
    ext<-GPUobject(ext, as.integer(X[2]),as.integer(X[2]))
    gc()
    return(ext)

}


