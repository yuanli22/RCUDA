#' scaleGPU
#'
#' This function scales the given vector by a scalar
#' by using CUDA cublas function cublasDcopy
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha scale factor
#' @return scaled vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- createGPU(a)
#' scaleGPU(a_gpu, b)->b_gpu
#' gatherGPU(b_gpu)


scaleGPU<-function(input, alpha)
{
    checkGPU(input)
    ext<-.Call(
                "scaleGPU",
                input$ptr,                        
                as.integer(input[2])*as.integer(input[3]),
                as.numeric(alpha)
              )
     ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)
}




#' sqrtGPU
#'
#' This function computes the square root of given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return square root vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' sqrtGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


sqrtGPU <- function (input) 
{ 

    checkGPU(input)
    ext<-.Call("vector_sqrt", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }




#' logGPU
#'
#' This function computes the natural logarithms of given vector or matrix
#' by using CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return natural logarithms vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' logGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


logGPU <- function (input) 
{
    checkGPU(input) 
    ext<-.Call("vector_log", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }




#' expGPU
#'
#' This function computes the exponential of given vector or matrix
#' by using CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return exponential vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{scaleGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' expGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)



expGPU <- function (input) 
{ 
    checkGPU(input)   
    ext<-.Call("vector_exp", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)
  }




#' powerGPU
#'
#' This function computes the power of given vector or matrix 
#' by using CUDA function 
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha power factor
#' @return powered vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{sqrtGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- createGPU(a)
#' powerGPU(a_gpu, b)->b_gpu
#' gatherGPU(b_gpu)


powerGPU <- function (input, alpha=1) 
{
    checkGPU(input)
    ext<-.Call("vector_power", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                as.numeric(alpha),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }
 


#' betaGPU
#'
#' This function apply the beta function to a given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return beta result, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' betaGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


betaGPU <- function (input1, input2) 
{ 
    checkGPU(input1)
    checkGPU(input2)
    if (as.integer(input1[2])*as.integer(input1[3])!=as.integer(input2[2])*as.integer(input2[3]))
    stop ("vectors dimension don't match")
    ext<-.Call("vector_beta", 
                input1$ptr,
                input2$ptr,
                as.integer(input1[2])*as.integer(input1[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input1[2]),as.integer(input1[3]))
    return(ext)

  }



#' gammaGPU
#'
#' This function apply the gammma function to a given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return gamma vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' gammaGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


gammaGPU <- function (input) 
{ 

    checkGPU(input)
    ext<-.Call("vector_gamma", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }

