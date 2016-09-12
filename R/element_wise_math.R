#' scalegpu
#'
#' This function scales the given vector/matrix by a scalar
#' by using CUDA cublas function cublasDcopy
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha scale factor
#' @return scaled vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- creategpu(a)
#' scalegpu(a_gpu, b) -> b_gpu
#' gathergpu(b_gpu)

scalegpu <- function(input, alpha)
{
    checkGPU(input)
    ext <- .Call(
                "scaleGPU",
                input$ptr,                        
                as.integer(input[2]) * as.integer(input[3]),
                as.numeric(alpha)
              )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
}


#' sqrtgpu
#'
#' This function computes the square root of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return square root of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' sqrtgpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

sqrtgpu <- function(input) 
{ 

    checkGPU(input)
    ext <- .Call("vector_sqrt", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }


#' loggpu
#'
#' This function computes the natural logarithms of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return natural logarithms of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' loggpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

loggpu <- function(input) 
{
    checkGPU(input) 
    ext <- .Call("vector_log", 
                input$ptr,
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }


#' expgpu
#'
#' This function computes the exponential of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return exponential of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{loggpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' expgpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

expgpu <- function(input) 
{ 
    checkGPU(input)   
    ext <- .Call("vector_exp", 
                input$ptr,
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }


#' powergpu
#'
#' This function computes the power of given vector/matrix 
#' by using self-defined CUDA function 
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha power factor
#' @return powered vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{sqrtgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- creategpu(a)
#' powergpu(a_gpu, b) -> b_gpu
#' gathergpu(b_gpu)

powergpu <- function(input, alpha = 1) 
{
    checkGPU(input)
    ext <- .Call("vector_power", 
                input$ptr,
                as.integer(input[2]) * as.integer(input[3]),
                as.numeric(alpha),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }
 

#' betagpu
#'
#' This function computes the beta function of the given vector/matrix
#' by using self-defined CUDA function
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension 
#' @return beta function result of given vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{gammagpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' betagpu(a_gpu, a_gpu) -> b_gpu
#' gathergpu(b_gpu)

betagpu <- function (x, y) 
{ 
    checkGPU(x)
    checkGPU(y)
    if (as.integer(x[2]) * as.integer(x[3])
        != as.integer(y[2]) * as.integer(y[3]))
    	stop ("vectors dimension don't match")
    ext <- .Call("vector_beta", 
                x$ptr,
                y$ptr,
                as.integer(x[2]) * as.integer(x[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
    return(ext)
  }


#' gammagpu
#'
#' This function computes the gammma function of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return gamma result of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{betagpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' gammagpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

gammagpu <- function(input) 
{ 
    checkGPU(input)
    ext <- .Call("vector_gamma", 
                input$ptr,
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }


#' dgammagpu
#'
#' This function computes the gammma pdf function of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @param k shape parameter of Gamma distribution; default value 1
#' @param theta scale parameter of Gamma distribution; default value 1  
#' @return gamma pdf result of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{dbetagpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' dgammagpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

dgammagpu <- function(input, k = 1, theta = 1) 
{ 
    checkGPU(input)
    ext <- .Call("vector_gammapdf", 
                input$ptr,
		  as.integer(input[2]) * as.integer(input[3]),
	         as.numeric(k),
	         as.numeric(theta),
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }


#' dbetagpu
#'
#' This function computes the beta pdf function of given vector/matrix
#' by using self-defined CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @param k shape parameter of Beta distribution; default value 1
#' @param theta scale parameter of Beta distribution; default value 1  
#' @return beta pdf result of vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{dbetagpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' dbetagpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

dbetagpu <- function(input, k = 1, theta = 1) 
{ 
    checkGPU(input)
    ext <- .Call("vector_betapdf", 
                input$ptr,
		  as.integer(input[2]) * as.integer(input[3]),
	         as.numeric(k),
	         as.numeric(theta),
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
               )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
  }
