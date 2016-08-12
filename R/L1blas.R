#' amaxgpu
#'
#' This function finds the (smallest) index of the element with 
#' the maximum magnitude of given vector/matrix
#' by using CUDA cublas function cublasIdamax
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{amingpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' amaxgpu(a_gpu)

amaxgpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                  "maxGPU",
                  input$ptr,              
                  as.integer(input[2]) * as.integer(input[3])
                )
    return(ext)
}


#' amingpu
#'
#' This function finds the (smallest) index of the element 
#' with the minimum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{amaxgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' amingpu(a_gpu) 

amingpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                  "minGPU",
                  input$ptr,              
                  as.integer(input[2]) * as.integer(input[3])
                 )
    return(ext)
}


#' asumgpu
#'
#' This function computes the summation 
#' of the elements' absolute values of given vector/matrix
#' by using CUDA cublas function cublasDasum
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the vector/matrix's elements absolute values summation 
#' @seealso \code{\link{amaxgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' asumgpu(a_gpu) 

asumgpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                  "asumGPU",
                  input$ptr,              
                  as.integer(input[2]) * as.integer(input[3])
                 )
    return(ext)
}


#' axpygpu
#'
#' This function multiplies the vector x by the scalar a and adds it 
#' to the vector y, and overwrites y as the result.  
#' by using CUDA cublas function cublasDaxpy. y = a x + y
#' @param x list consisting of R external GPU pointer and dimension
#' @param y list consisting of R external GPU pointer and dimension 
#' @param alpha scale factor alpha; default 1
#' @return updated y vector/matrix
#' @seealso \code{\link{scalgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(a)
#' axpygpu(a_gpu, b_gpu, 1) 

axpygpu<-function(x, y, alpha = 1)
{
    checkGPU(x)
    checkGPU(y)
    if (as.integer(x[2]) * as.integer(x[3])
        != as.integer(y[2]) * as.integer(y[3]))
     stop ("vectors dimension don't match")
    ext <- .Call(
                  "axpyGPU",
                  x$ptr,  
                  y$ptr,             
                  as.integer(x[2]) * as.integer(x[3]),
                  alpha 
                 )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
    return(ext)
}


#' copygpu
#'
#' This function copies the vector x into the vector y  
#' by using CUDA cublas function cublasDcopy
#' @param x list consisting of R external GPU pointer and dimension
#' @param y list consisting of R external GPU pointer and dimension
#' @return copied vector/matrix
#' @seealso \code{\link{axpygpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' copygpu(a_gpu, b_gpu) 

copygpu<-function(x, y)
{
    checkGPU(x)
    checkGPU(y)
    if (as.integer(x[2]) * as.integer(x[3])
        != as.integer(y[2]) * as.integer(y[3]))
     stop ("vectors dimension don't match")
    ext <- .Call(
                  "copyGPU",
                  x$ptr,   
                  y$ptr,          
                  as.integer(x[2]) * as.integer(x[3])
                 )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
    return(ext)
}


#' scalgpu
#'
#' This function scales the vector x by the scalar a  
#' and overwrites it with the result 
#' by using CUDA cublas function cublasDscal
#' @param x list consisting of R external GPU pointer and dimension
#' @param alpha scale factor alpha, default 1
#' @return scaled vector/matrix
#' @seealso \code{\link{scalegpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' scalgpu(a_gpu, 2) 

scalgpu<-function(x, alpha = 1)
{
    checkGPU(x)
    ext <- .Call(
                  "scalGPU",
                  x$ptr,            
                  as.integer(x[2]) * as.integer(x[3]),
                  alpha 
                 )
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
    return(ext)
}


#' dotgpu
#'
#' This function computes the dot product of two given vectors/matrix
#' by using CUDA cublas function cublasDdot
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return the resulting dot product 
#' @seealso \code{\link{nrm2gpu}} 
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


#' nrm2gpu
#'
#' This function computes Euclidean norm of given 
#' vector/matrix by using CUDA cublas function cublasDnrm2
#' @param input list consisting of R external GPU pointer and dimension 
#' @return vector Euclidean norm, a non-negative number
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{dotgpu}}  
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' nrm2gpu(a_gpu) 

nrm2gpu<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                   "norm2GPU",
                   input$ptr,              
                   as.integer(input[2]) * as.integer(input[3])
                 )
    return(ext)
}


