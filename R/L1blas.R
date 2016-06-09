#' maxGPU
#'
#' finds the (smallest) index of the element with the maximum magnitude of given vector
#' This function finds the (smallest) index of the element with the maximum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{minGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' maxGPU(a_gpu)

maxGPU<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                "maxGPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
              )
    return(ext)
}


#' minGPU
#'
#' finds the (smallest) index of the element with the minimum magnitude of given vector
#' This function finds the (smallest) index of the element with the minimum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{maxGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' minGPU(a_gpu) 


minGPU<-function(input)
{
    checkGPU(input)
    ext<-.Call(
                "minGPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
              )
    return(ext)

}


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


#' norm2GPU
#'
#' Compute the Euclidean norm of given vector 
#' 
#' This function computes Euclidean norm of given vector by using CUDA cublas function cublasDnrm2
#' @param input list consisting of R external GPU pointer and dimension 
#' @return vector Euclidean norm, a non-negative number.
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' norm2GPU(a_gpu) 


norm2GPU<-function(input)
{
    checkGPU(input)
    ext<-.Call("norm2GPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
               )
    return(ext)

}


