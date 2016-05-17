checkGPU<-function(input)
{
    if (sum(class(input)=="GPUvector")==0)
    stop("input is not a GPUvector")
}
