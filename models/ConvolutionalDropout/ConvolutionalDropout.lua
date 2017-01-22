function hanning(n)
    return torch.range(0,n-1):mul(2*math.pi):div(n-1):cos():mul(-1):add(1):div(2)
end

function hanning2(n)
    local t = hanning(n):reshape(n,1)
    return torch.cmul(t:expand(n,n), t:expand(n,n):t())
end

function hanning2mask(n)
    return -hanning2(n) + 1
end

function conv_dropout(n, chan, imsize, masksize, img, pattern)
    if img == nil then
      img = torch.ones(chan, imsize, imsize)
    end
    if pattern == nil then
      pattern = hanning2mask(masksize):reshape(1, masksize, masksize):expand(chan, masksize, masksize)
    end

    for i = 1,n do
        local row = torch.random(1, imsize - masksize + 1)
        local col = torch.random(1, imsize - masksize + 1)
        img[{ {},{row,row + masksize - 1},{col, col + masksize - 1} }]:cmul(pattern)
    end
    return img
end


local ConvolutionalDropout, parent = torch.class('nn.ConvolutionalDropout', 'nn.Module')

function ConvolutionalDropout:__init(n, masksize)
  parent.__init(self)
  self.masksize = masksize
  self.n = n
  self.pattern = hanning2mask(masksize) -- :reshape(1,masksize,masksize)
  self.mask = torch.Tensor()
end

function ConvolutionalDropout:updateOutput(input)
  local bs = input:size(1)
  local ch = input:size(2)
  local h = input:size(3)
  local w = input:size(4)
  -- generate mask
  self.mask:resize(bs,1,h,w):fill(1)
  self.noise = self.mask:expand(bs,ch,h,w)
  for i = 1,bs do
    for j = 1,self.n do
      local row = torch.random(1, h - self.masksize + 1)
      local col = torch.random(1, w - self.masksize + 1)
      self.mask[{ i,1,{row,row+self.masksize-1},{col,col+self.masksize-1} }]:cmul(self.pattern)
    end
  end
  -- compute output
  self.output:resizeAs(input):copy(input):cmul(self.noise)
  return self.output
end

function ConvolutionalDropout:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  self.gradInput:cmul(self.noise)
  return self.gradInput
end

function ConvolutionalDropout:__tostring__()
   return string.format('%s(%d, %d)', torch.type(self), self.n, self.masksize)
end

function ConvolutionalDropout:clearState()
   if self.mask then
      self.mask:set()
      self.pattern:set()
   end
   return Parent.clearState(self)
end
