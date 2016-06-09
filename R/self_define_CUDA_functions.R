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



#' sumGPU
#'
#' Compute the sum of given vector  
#' 
#' This function computes sum of given vector by using CUDA vector reduction
#' @param x list consisting of R external GPU pointer and dimension 
#' @return vector sum
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- createGPU(1:4)
#' sumGPU(a)

sumGPU<-function(x)
{
 checkGPU(x) 
 ext<-.Call(
                "vector_reduction",
                x$ptr,      
                as.integer(x[2])*as.integer(x[3]),
                PACKAGE= "supplement" 
              )
 
    return(ext)

}


#' varGPU
#'
#' Compute the variance of given sample  
#' 
#' This function computes variance of given sample by using CUDA vector reduction
#' @param x list consisting of R external GPU pointer and dimension 
#' @return sample variance
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{sumGPU}}
#' @export
#' @examples
#' a <- createGPU(1:4)
#' varGPU(a)

varGPU<-function(x)
{
 checkGPU(x) 
 n <- as.integer(x[2])*as.integer(x[3])
 mean <- sumGPU(x)/n
 ext <- .Call(
                "cudavarGPU",
                x$ptr, 
                n,
                as.numeric(mean),
                PACKAGE= "supplement" 
              )
 ext<-GPUobject(ext, as.integer(x[2]),as.integer(x[3]))
 result <- sumGPU(ext)/(n-1) 
 return(result)

}



#' subsetGPU
#'
#' This function select and copy subset of a given GPU vector
#' by using CUDA function
#' @param input list consisting of R external GPU pointer and dimension
#' @param index index of the vector subset 
#' @return subset of vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' subsetGPU(a_gpu,c(1,2))->b_gpu
#' gatherGPU(b_gpu)



subsetGPU <- function (input, index) 
{ 
    checkGPU(input)
    n <- length(index) 
    index <- as.integer(index)

    if ((as.integer(input[2])*as.integer(input[3])<max(index))|(min(index)<1))
    stop ("index out of bound")

    ext <- .Call("subset_GPU", 
                input$ptr,
                as.integer(n),
                as.integer(index),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(n),1)
    return(ext)
  }
