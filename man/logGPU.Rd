% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/element_wise_math.R
\name{loggpu}
\alias{loggpu}
\title{loggpu}
\usage{
loggpu(input)
}
\arguments{
\item{input}{list consisting of R external GPU pointer and dimension}
}
\value{
natural logarithms of vector/matrix, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{number of rows}
\item{n: }{number of columns}
}
}
\description{
This function computes the natural logarithms of given vector/matrix
by using self-defined CUDA function
}
\examples{
a <- 1:4
a_gpu <- creategpu(a)
loggpu(a_gpu) -> b_gpu
gathergpu(b_gpu)
}
\seealso{
\code{\link{expgpu}}
}

