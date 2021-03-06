% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/self_define_CUDA_functions.R
\name{subsetgpu}
\alias{subsetgpu}
\title{subsetgpu}
\usage{
subsetgpu(input, index)
}
\arguments{
\item{input}{list consisting of R external GPU pointer and dimension}

\item{index}{index of the vector/matrix subset}
}
\value{
subset of the given vector/matrix, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{number of rows}
\item{n: }{number of columns}
}
}
\description{
This function returns the specified subset of given GPU vector/matrix
by using self-defined CUDA function
}
\examples{
a <- 1:4
a_gpu <- creategpu(a)
subsetgpu(a_gpu,c(1, 2))->b_gpu
gathergpu(b_gpu)
}
\seealso{
\code{\link{creategpu}}
}

