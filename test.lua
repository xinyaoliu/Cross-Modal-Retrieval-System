require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'

local utils = require 'utils.misc'


cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-picture', '/home/liuxinyao/dataset/coco/2014/images/train2014/COCO_train2014_000000044474.jpg', '/path/to')
opt = cmd:parse(arg or {})

local ok, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not ok then print('package cunn not found!') end
if not ok2 then print('package cutorch not found!') end
if ok and ok2 then
    
    cutorch.setDevice(2 + 1) -- torch is 1-indexed
    cutorch.manualSeed(981723)
else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back to CPU mode')
    -- opt.gpuid = -1
end

a = torch.load('/home/liuxinyao/caffemodel/nin/nin_in_torch/logs/nin/model.net','binary')

function predict( pic )
	-- body
	inputs = utils.preprocess(image.scale(image.load(opt.picture, 3), 224, 224))
	n_input:set(1,inputs)
	n_input = n_input:cuda()
	output = a:forward(1,n_input)
	print (output)
end

predict(opt.picture)
