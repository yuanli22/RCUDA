% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/random_number_genertor.R
\name{rlognormgpu}
\alias{rlognormgpu}
\title{rlognormgpu}
\usage{
rlognormgpu(n, mean = 0, sd = 1, seed = 1)
}
\arguments{
\item{n}{number of random numbers}

\item{mean}{mean of log-normal distribution; default value 0}

\item{sd}{standard deviation of log-normal distribution; default value 1}

\item{seed}{random number generator seed; default value 1}
}
\value{
generated random numbers vector, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{number of rows}
\item{n: }{number of columns}
}
}
\description{
This function generates log-normally distributed random numbers 
by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT 
and curandGenerateLogNormalDouble
}
\examples{
a_gpu <- rlognormgpu(100, 0, 1, 15)
gathergpu(a_gpu)
}
\seealso{
\code{\link{rnormgpu}}
}

