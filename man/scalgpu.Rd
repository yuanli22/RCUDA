% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/L1blas.R
\name{scalgpu}
\alias{scalgpu}
\title{scalgpu}
\usage{
scalgpu(x, alpha = 1)
}
\arguments{
\item{x}{list consisting of R external GPU pointer and dimension}

\item{alpha}{scale factor alpha, default 1}
}
\value{
scaled vector/matrix
}
\description{
This function scales the vector x by the scalar a  
and overwrites it with the result 
by using CUDA cublas function cublasDscal
}
\examples{
a <- 1:4
a_gpu <- creategpu(a)
scalgpu(a_gpu, 2) 
}
\seealso{
\code{\link{scalegpu}}
}

