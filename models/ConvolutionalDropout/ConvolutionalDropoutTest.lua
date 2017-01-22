require 'nn'
require 'gnuplot'
require 'image'
require 'ConvolutionalDropout'


n = nn.ConvolutionalDropout(5,30)
n:forward(torch.rand(10,3,300,300))
image.display(n.output[1])
image.display(n.output[2])
image.display(n.output[3])
