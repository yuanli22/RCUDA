#' maxgpu
#'
#' This function finds the (smallest) index of the element with 
#' the maximum magnitude of given vector/matrix
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{mingpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' maxgpu(a_gpu)

maxgpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                  "maxGPU",
                  input$ptr,              
                  as.integer(input[2]) * as.integer(input[3])
                )
    return(ext)
}


#' mingpu
#'
#' This function finds the (smallest) index of the element 
#' with the minimum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{maxgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' mingpu(a_gpu) 

mingpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                  "minGPU",
                  input$ptr,              
                  as.integer(input[2]) * as.integer(input[3])
                 )
    return(ext)
}


#' dotgpu
#'
#' This function computes the dot product of two given vectors/matrix
#' by using CUDA cublas function cublasDdot
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return the resulting dot product 
#' @seealso \code{\link{norm2gpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' dotgpu(a_gpu, b_gpu)

dotgpu <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      != as.integer(y[2]) * as.integer(y[3]))
     stop ("vectors dimension don't match")
  ext <- .Call(
                "dotGPU",
                 x$ptr,
                 y$ptr,
                 as.integer(x[2]) * as.integer(x[3])
               )
   return(ext)
}


#' norm2gpu
#'
#' This function computes Euclidean norm of given 
#' vector/matrix by using CUDA cublas function cublasDnrm2
#' @param input list consisting of R external GPU pointer and dimension 
#' @return vector Euclidean norm, a non-negative number
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gathergpu}}  
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' norm2gpu(a_gpu) 

norm2gpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                   "norm2GPU",
                   input$ptr,              
                   as.integer(input[2]) * as.integer(input[3])
                 )
    return(ext)
}


