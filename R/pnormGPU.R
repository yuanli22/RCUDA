#' pnormGPU
#'
#' This function computing the standard normal cumulative density function input vector
#' @param input list consisting of R external GPU pointer and dimension 
#' @return standard normal CDF result, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{qnormGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' pnormGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


pnormGPU<-function(input)
{
    checkGPU(input)
    ext<-.Call(
                "cudanormalCDF",
                input$ptr,           
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"
              )
     ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)
}