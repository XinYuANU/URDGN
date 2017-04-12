------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'adverserial_xin_v1_D'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs512_lfw64")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 64)          scale of images to train on
  --lambda           (default 0.01)       trade off D and Euclidean distance 
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

ntrain = 14720
nval = 768

local highHd5 = hdf5.open('datasets/YTC_HR.hdf5', 'r')
local data_HR = highHd5:read('YTC'):all()
data_HR:mul(2):add(-1)
highHd5:close()

--local num_data = torch.Tensor{data_HR:size()[1]}
--ntrain = torch.floor(torch.mul(num_data, 0.95))
--ntrain = ntrain - torch.mod(ntrain, opt.batchSize)
--nval   = num_data - ntrain
--nval   = nval - torch.mod(nval, opt.batchSize)
--print(num_data, ntrain+nval)

trainData_HR = data_HR[{{1, ntrain}}]
valData_HR = data_HR[{{ntrain+1, nval+ntrain}}]

local lowHd5 = hdf5.open('datasets/YTC_LR.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+1, nval+ntrain}}]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  model_D = nn.Sequential()
  model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))  
  model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(128, 96, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(nn.Reshape(8*8*96))
  model_D:add(nn.Linear(8*8*96, 1024))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(1024,1))
  model_D:add(nn.Sigmoid())

  model_G = nn.Sequential()
  model_G:add(cudnn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
  model_G:add(nn.SpatialBatchNormalization(16))
  model_G:add(cudnn.ReLU(true))  
  model_G:add(nn.SpatialUpSamplingNearest(2))  
  model_G:add(cudnn.SpatialConvolution(16, 64, 3, 3, 1, 1, 1, 1))
  model_G:add(nn.SpatialBatchNormalization(64))
  model_G:add(cudnn.ReLU(true))  
  model_G:add(nn.SpatialUpSamplingNearest(2))
  model_G:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  model_G:add(nn.SpatialBatchNormalization(128))
  model_G:add(cudnn.ReLU(true))
  model_G:add(nn.SpatialUpSamplingNearest(2))
  model_G:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2))  
  model_G:add(nn.SpatialBatchNormalization(256))
  model_G:add(cudnn.ReLU(true))

  model_G:add(cudnn.SpatialConvolution(256,3, 5, 5, 1, 1, 2, 2))

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion()
criterion_G = nn.MSECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)


-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

-- Get examples to plot
function getSamples(dataset, N)
  local numperclass = numperclass or 10
  local N = N or 8
  
--  local noise_inputs = torch.Tensor(N, opt.noiseDim)
  -- print(nval:long(),nval)
 
  local noise_input_high = dataset
  local noise_inputs = torch.Tensor(N, 3, 16, 16)
  for i = 1,N do
	idx = math.random(nval)  
    noise_inputs[{{i}}] = image.scale(torch.squeeze(noise_input_high[{{idx}}]),16,16) 
  end
  -- Generate samples
  
  -- noise_inputs:normal(0, 1)
  -- print(noise_inputs:size())  
  local samples = model_G:forward(noise_inputs)
  --print(samples:size())  
  samples = nn.HardTanh():forward(samples)
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = samples[i]:float()
  end
  
  return to_plot
end


-- training loop
while true do
  local to_plot = getSamples(valData_HR, 100)  
  torch.setdefaulttensortype('torch.FloatTensor')

  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
  --testLogger:style{['% mean class accuracy (test set)'] = '-'}
  trainLogger:plot()
  --testLogger:plot()

  local formatted = image.toDisplayTensor({input=to_plot, nrow=10})
  formatted:float()
  formatted = formatted:index(1,torch.LongTensor{3,2,1})
  
  image.save(opt.save .."/YTC_example_v1_"..(epoch or 0)..'.png', formatted)
  
  print(nval+ntrain)  
  IDX = torch.randperm(14720)    
  
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end


  -- train/test
  adversarial.train(trainData_LR,trainData_HR)
--  adversarial.test(valData_LR,valData_HR)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)


end
