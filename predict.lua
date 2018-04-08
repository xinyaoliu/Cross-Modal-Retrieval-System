require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'
local DataLoader2 = require 'utils.DataLoaderForTest'


require 'loadcaffe'
local LSTM = require 'LSTM'
-- local GRU = require 'GRU'
-- local RNN = require 'RNN'

cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-rnn_size', 512, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'Number of layers in LSTM')
cmd:option('-embedding_size', 512, 'size of word embeddings')
-- optimization
cmd:option('-batch_size', 50, 'batch size')
-- bookkeeping
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-checkpoint_file', 'checkpoints/bs_epoch44.49_0.0000.t7', 'Checkpoint file to use for predictions')
cmd:option('-data_dir', '/home/liuxinyao/bs/code', 'data directory')
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-feat_layer', 'fc7', 'Layer to extract features from')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
cmd:option('-input_image_path', 'data/train2014/COCO_train2014_000000405541.jpg', 'Image path')
cmd:option('-caption', 'Something in the bathroom', 'Captionn string')
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
        cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back to CPU mode')
        opt.gpuid = -1
    end
end

loader = DataLoader.create(opt.data_dir, opt.batch_size, opt, 'predict')
loader2 = DataLoader2.create(opt.data_dir, opt.batch_size, opt)

-- load model checkpoint

print('loading checkpoint from ' .. opt.checkpoint_file)
checkpoint = torch.load(opt.checkpoint_file)
-- print(checkpoint)
print("checkpoint end")

lstm_clones = {}
lstm_clones = utils.clone_many_times(checkpoint.protos.lstm, loader.c_max_length + 1)

checkpoint.protos.ltw:evaluate()
-- checkpoint.protos.lti:evaluate()

c_vocab_size = checkpoint.vocab_size


c_iv = {}
for i,v in pairs(loader.c_vocab_mapping) do
    c_iv[v] =  i
end

if c_vocab_size ~= loader.c_vocab_size then
    print('Vocab size of checkpoint and data are different.')
end

-- cnn = loadcaffe.load(opt.proto_file, opt.model_file)

lti = nn.Sequential()
lti:add(nn.Linear(4096, opt.embedding_size))
lti:add(nn.Tanh())
lti:add(nn.Dropout(opt.dropout))


function predict(caption_string)

    -- extract image features

    if opt.gpuid >= 0 then
        -- cnn = cnn:cuda()
        lti = lti:cuda()
    end

   
-- todo
    -- local img = utils.preprocess(image.scale(image.load(input_image_path, 3), 224, 224))

    i_batch, id_batch = loader2:next_batch_for_test()
    imf = lti:forward(i_batch)  

    if opt.gpuid >= 0 then
        imf = imf:cuda()
    end

-- ----------------

    -- 1264025
    -- img_batch = torch.zeros(10000, 3, 224, 224)
    -- for i = 1, 10000 do
    --     local fp = path.join(opt.input_image_dir, string.format('%s2014/COCO_%s2014_%.12d.jpg', opt.split, opt.split, image_id[idx]))
    --     img_batch[i] = utils.preprocess(image.scale(image.load(fp, 3), 224, 224))

    -- todo
    -- local fc7 = cnn_fc7:forward(img)
    -- local imf = lti:forward(fc7)

    -- extract question features


-- --------------

    local caption = torch.ShortTensor(loader.c_max_length):zero()

    local idx = 1
    local words = {}
    for token in string.gmatch(caption_string, "%a+") do
        words[idx] = token
        idx = idx + 1
    end

    for i = 1, #words do
        caption[loader.c_max_length - #words + i] = loader.c_vocab_mapping[words[i]] or loader.c_vocab_mapping['UNK']
    end

    if opt.gpuid >= 0 then
        caption = caption:cuda()
    end

    -- 1st index of `nn.LookupTable` is for zeros
    caption = caption + 1

    local qf = checkpoint.protos.ltw:forward(caption)

    -- lstm + softmax

    local init_state = {}
    for L = 1, opt.num_layers do
        local h_init = torch.zeros(1, opt.rnn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(init_state, h_init:clone())
        if opt.model == 'lstm' then
            table.insert(init_state, h_init:clone())
        end
    end

    local rnn_state = {[0] = init_state}

    for t = 1, loader.c_max_length do
        lst = lstm_clones[t]:forward{qf:select(1,t):view(1,-1), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    -- local lst = lstm_clones[loader.q_max_length + 1]:forward{imf:view(1,-1), unpack(rnn_state[loader.q_max_length])}

    -- local prediction = checkpoint.protos.sm:forward(lst[#lst])

    prediction = lst[#lst]

    rprediction = torch.repeatTensor(prediction,4000,1)

    if opt.gpuid >= 0 then
        rprediction = rprediction:cuda()
    end

    print (rprediction:size())

    sp = prediction:storage()

    for i=1, sp:size() do
        if sp[i] <= 0 then
            sp[i] = 0
        else
            sp[i] = 1
        end
    end



    sf = imf:storage()
    
    for i=1, sf:size() do
        if sf[i] <= 0 then
            sf[i] = 0
        else
            sf[i] = 1
        end
    end



    mlp = nn.PairwiseDistance(1)
    
    if opt.gpuid >= 0 then
        mlp = mlp:cuda()
    end

    mresult = mlp:forward({rprediction,imf})

    result, rindex = torch.sort(mresult)
    print (result:narrow(1,1,10))
    print (rindex:narrow(1,1,10))
    -- print (id_batch[1377])
    print (id_batch[rindex[1]])
    print (id_batch[rindex[2]])
    print (id_batch[rindex[3]])
    print (id_batch[rindex[4]])
    print (id_batch[rindex[5]])
    print (id_batch[rindex[6]])
    print (id_batch[rindex[7]])
    print (id_batch[rindex[8]])
    print (id_batch[rindex[9]])
    print (id_batch[rindex[10]])

    -- local _, idx  = prediction:max(2)

    -- print(a_iv[idx[1][1]])
end

predict(opt.caption)

