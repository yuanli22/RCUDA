#' dnormgpu
#'
#' This function computes the normal distribution density of given vector/matrix
#'
#' @param input list consisting of R external GPU pointer and dimension 
#' @param mean vector/matrix of mean
#' @param sd vector/matrix of standard deviation 
#' @details If mean or sd are not specified they assume the default values
#' of 0 and 1, respectively.
#' @return normal distribution density vector/matrix, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{pnormgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' dnormgpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

dnormgpu <- function(input, mean=0, sd=1)
{
    checkGPU(input)
    ext <- .Call(
                "cudanormaldensity",
                input$ptr,            
                as.integer(input[2]) * as.integer(input[3]),
                as.numeric(mean),
                as.numeric(sd),
                PACKAGE = "supplement"
              )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
}


#' pnormgpu
#'
#' This function computes the standard normal distribution cumulative 
#' density (CDF) of given vector/matrix
#' 
#' @param input list consisting of R external GPU pointer and dimension 
#' @return standard normal CDF, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{dnormgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- creategpu(a)
#' pnormgpu(a_gpu) -> b_gpu
#' gathergpu(b_gpu)

pnormgpu <- function(input)
{
    checkGPU(input)
    ext <- .Call(
                "cudanormalCDF",
                input$ptr,           
                as.integer(input[2]) * as.integer(input[3]),
                PACKAGE = "supplement"
              )
    ext <- GPUobject(ext, as.integer(input[2]), as.integer(input[3]))
    return(ext)
}