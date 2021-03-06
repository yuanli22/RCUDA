% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/element_wise_math.R
\name{scalegpu}
\alias{scalegpu}
\title{scalegpu}
\usage{
scalegpu(input, alpha)
}
\arguments{
\item{input}{list consisting of R external GPU pointer and dimension}

\item{alpha}{scale factor}
}
\value{
scaled vector/matrix, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{number of rows}
\item{n: }{number of columns}
}
}
\description{
This function scales the given vector/matrix by a scalar
by using CUDA cublas function cublasDcopy
}
\examples{
a <- 1:4
b <- 2
a_gpu <- creategpu(a)
scalegpu(a_gpu, b) -> b_gpu
gathergpu(b_gpu)
}
\seealso{
\code{\link{expgpu}}
}

