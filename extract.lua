
require 'torch'
require 'nn'
require 'image'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoaderForTest'

require 'loadcaffe'

cmd = torch.CmdLine()
cmd:text('Options')

cmd:option('-batch_size', 10, 'batch size')
cmd:option('-split', 'test', 'train/val')
cmd:option('-debug', 0, 'set debug = 1 for lots of prints')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-data_dir', '/home/liuxinyao/bs/code', 'Data directory.')
cmd:option('-feat_layer', 'drop7', 'Layer to extract features from')
cmd:option('-input_image_dir', '/home/liuxinyao/dataset/coco/2014/images', 'Image directory')
-- gpu/cpu
cmd:option('-gpuid', 0, '0-indexed id of GPU to use. -1 = CPU')

opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)



if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1)
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back to CPU mode')
        opt.gpuid = -1
    end
end

loader = DataLoader.create(opt.data_dir, opt.batch_size, opt, 'fc7_feat')



------------------------------------------------------------------




tmp_image_id = {}

-- print (loader.data)

for i = 1, 10000 do

    tmp_image_id[loader.data['train'][i].image_id] = 1
end

image_id = {}
idx = 1
for i, v in pairs(tmp_image_id) do
    image_id[idx] = i
    idx = idx + 1
end

fc7 = torch.DoubleTensor(#image_id, 4096)
idx = 1

if opt.gpuid >= 0 then
    fc7 = fc7:cuda()
end

img_batch = torch.zeros(10000, 3, 224, 224)
img_id_batch = {}
for i = 1, 10000 do
    if not image_id[idx] then
        break
    end
    local fp = path.join(opt.input_image_dir, string.format('%s2014/COCO_%s2014_%.12d.jpg', 'test', 'test', image_id[idx]))
    if opt.debug == 1 then
        print(idx)
        print(fp)
    end
    img_batch[i] = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))
    img_id_batch[i] = image_id[idx]
    idx = idx + 1
end

if opt.gpuid >= 0 then
    img_batch = img_batch:cuda()
end


trainData = {
      data = img_batch,
      labels = trlabels,
      size = function() return 10000 end
   }


------------------------------------------------------------------




cnn = loadcaffe.load(opt.proto_file, opt.model_file)
if opt.gpuid >= 0 then
    cnn = cnn:cuda()
end

cnn_fc7 = nn.Sequential()

for i = 1, #cnn.modules do
    local layer = cnn:get(i)
    local name = layer.name
    cnn_fc7:add(layer)
    if name == opt.feat_layer then
        break
    end
end

cnn_fc7:add(nn.Linear(4096, 80))
cnn_fc7:add(nn.SoftMax())

print (cnn_fc7)

criterion = nn.ClassNLLCriterion()

-- cnn_fc7:evaluate()

if opt.gpuid >= 0 then
    cnn_fc7 = cnn_fc7:cuda()
    criterion = criterion:cuda()
end


classes = {'person',' bicycle ','car',' motorcycle ', 'airplane', 'bus',' train ',' truck ',' boat ',' traffic light', ‘fire hydrant’, ‘stop sign’, ‘parking meter’, ‘bench’, ‘bird’, ‘cat’, ‘dog’, ‘horse’, ‘sheep’, ‘cow’, ‘elephant’, ‘bear’, ‘zebra’, ‘giraffe’, ‘backpack’, ‘umbrella’, ‘handbag’, ‘tie’, ‘suitcase’, ‘frisbee’, ‘skis’, ‘snowboard’, ‘sports ball’, ‘kite’, ‘baseball bat’, ‘baseball glove’, ‘skateboard’, ‘surfboard’, ‘tennis racket’, ‘bottle’, ‘wine glass’, ‘cup’, ‘fork’, ‘knife’, ‘spoon’, ‘bowl’, ‘banana’, ‘apple’, ‘sandwich’, ‘orange’, ‘broccoli’, ‘carrot’, ‘hot dog’, ‘pizza’, ‘donut’, ‘cake’, ‘chair’, ‘couch’, ‘potted plant’, ‘bed’, ‘dining table’, ‘toilet’, ‘tv’, ‘laptop’, ‘mouse’, ‘remote’, ‘keyboard’, ‘cell phone’, ‘microwave’, ‘oven’, ‘toaster’, ‘sink’, ‘regrigerator’, ‘book’, ‘clock’, ‘vase’, ‘scissors’, ‘teddy bear’, ‘hair drier’, ‘toothbrush’}
confusion = optim.ConfusionMatrix(classes)


-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log')) --zaiyi
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


if cnn_fc7 then
   parameters,gradParameters = cnn_fc7:getParameters()
end


learningRate = 1e-3
weightDecay = 0
momentum = 0
learningRateDecay = 1e-7



------------------------------------------------------------------





function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   cnn_fc7:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize) --zaiyi

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')





   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      -- xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
        local input = trainData.data[shuffle[i]]
        local target = trainData.labels[shuffle[i]]
         
        input = input:cuda()
        if opt.loss == 'mse' then
           target = target:cuda()
        end
         
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = cnn_fc7:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          cnn_fc7:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')  --zaiyi
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, cnn_fc7)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end



------------------------------------------------------------------




function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   cnn_fc7:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      -- xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      -- if opt.type == 'double' then input = input:double()
      input = input:cuda()
      local target = testData.labels[t]

      -- test sample
      local pred = cnn_fc7:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end



print '==> training!'

while true do
   train()
   test()
end




-- tmp_image_id = {}

-- -- print (loader.data)

-- for i = 1, 40775 do

--     tmp_image_id[loader.data['test'][i].image_id] = 1
-- end

-- image_id = {}
-- idx = 1
-- for i, v in pairs(tmp_image_id) do
--     image_id[idx] = i
--     idx = idx + 1
-- end

-- fc7 = torch.DoubleTensor(#image_id, 4096)
-- idx = 1

-- if opt.gpuid >= 0 then
--     fc7 = fc7:cuda()
-- end

repeat
    local timer = torch.Timer()
    -- img_batch = torch.zeros(opt.batch_size, 3, 224, 224)
    -- img_id_batch = {}
    -- for i = 1, opt.batch_size do
    --     if not image_id[idx] then
    --         break
    --     end
    --     local fp = path.join(opt.input_image_dir, string.format('%s2014/COCO_%s2014_%.12d.jpg', 'test', 'test', image_id[idx]))
    --     if opt.debug == 1 then
    --         print(idx)
    --         print(fp)
    --     end
    --     img_batch[i] = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))
    --     img_id_batch[i] = image_id[idx]
    --     idx = idx + 1
    -- end

    -- if opt.gpuid >= 0 then
    --     img_batch = img_batch:cuda()
    -- end

    fc7_batch = cnn_fc7:forward(img_batch:narrow(1, 1, #img_id_batch))

    for i = 1, fc7_batch:size(1) do
        if opt.debug == 1 then
            print(idx - fc7_batch:size(1) + i - 1)
        end
        fc7[idx - fc7_batch:size(1) + i - 1]:copy(fc7_batch[i])
    end

    if opt.gpuid >= 0 then
        cutorch.synchronize()
    end

    local time = timer:time().real
    print(idx-1 .. '/' .. #image_id .. " " .. time)
    collectgarbage()
until idx >= #image_id

torch.save(path.join(opt.data_dir, opt.split .. '_fc7.t7'), fc7)
torch.save(path.join(opt.data_dir, opt.split .. '_fc7_image_id.t7'), image_id)
