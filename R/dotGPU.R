#' dotGPU
#'
#' This function computes the dot product of two given vectors
#' by using CUDA cublas function cublasDdot
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return the resulting dot product 
#' @seealso \code{\link{norm2GPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- createGPU(a)
#' b_gpu <- createGPU(b)
#' dotGPU(a_gpu, b_gpu)

dotGPU <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2])*as.integer(x[3])!=as.integer(y[2])*as.integer(y[3]))
  stop ("vectors dimension don't match")
  ext <- .Call(
              "dotGPU",
               x$ptr,
               y$ptr,
               as.integer(x[2])*as.integer(x[3])
            )
 
   return(ext)
}
