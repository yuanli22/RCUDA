#' divideGPU
#'
#' This function computes the element-wise division of two given vectors or matrices (x / y)
#' by using CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise division of vectors or matrices (x / y), a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{multiplyGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- createGPU(a)
#' b_gpu <- createGPU(b)
#' divideGPU(a_gpu, b_gpu)->c_gpu
#' gatherGPU(c_gpu)


divideGPU <- function (x, y) 
{ 

  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2])*as.integer(x[3])!=as.integer(y[2])*as.integer(y[3]))
  stop ("vectors dimension don't match")
    ext<-.Call("vector_divide", 
                x$ptr,
                y$ptr, 
                as.integer(x[2])*as.integer(x[3]),
                PACKAGE= "supplement"

               )
  if (as.integer(x[3])!=1)
    {ext<-GPUobject(ext, as.integer(x[2]),as.integer(x[3]))}
  else
    {ext<-GPUobject(ext, as.integer(y[2]),as.integer(y[3]))}
    return(ext)
  }

