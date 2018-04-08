
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local utils = require 'utils.misc'
local DataLoader = require 'utils.DataLoader'

local LSTM = require 'LSTM'
-- local GRU = require 'GRU'
-- local RNN = require 'RNN'

cmd = torch.CmdLine()
cmd:text('Options')

-- model params
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-rnn_size', 52, 'Size of LSTM internal state')
cmd:option('-num_layers', 2, 'Number of layers in LSTM')
cmd:option('-embedding_size', 52, 'Size of word embeddings')
-- optimization
cmd:option('-learning_rate', 1e-5, 'Learning rate')
cmd:option('-learning_rate_decay', 0.9, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 15, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-alpha', 0.8, 'alpha for adam')
cmd:option('-beta', 0.999, 'beta used for adam')
cmd:option('-epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
cmd:option('-batch_size', 50, 'Batch size')
cmd:option('-max_epochs', 50, 'Number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 1000, 'No. of iterations after which to checkpoint')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
cmd:option('-data_dir', '/home/liuxinyao/bs/code', 'Data directory')
cmd:option('-checkpoint_dir', '/home/liuxinyao/bs/code/checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'bs', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', 2, '0-indexed id of GPU to use. -1 = CPU')

-- parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)
torch.manualSeed(opt.seed)

-- gpu stuff
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

-- initialize the data loader
-- checks if .t7 data files exist
-- if they don't or if they're old,
-- they're created from scratch and loaded
local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt)


-- create the directory for saving snapshots of model at different times during training
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = true
if string.len(opt.init_from) > 0 then

    -- initializing model from checkpoint
    print('Loading model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    do_random_init = false
else
    -- model definition  
    -- components: ltw, lti, lstm and sm
    protos = {}
    protos2 = {}
    -- ltw: lookup table + dropout for caption words
    -- each word of the caption gets mapped to its index in vocabulary
    -- and then is passed through ltw to get a vector of size `embedding_size`
    -- lookup table dimensions are `vocab_size` x `embedding_size`
    protos.ltw = nn.Sequential()
    print ('c_vocab_size is ' .. loader.c_vocab_size .. 'embedding_size is ' .. opt.embedding_size)
    protos.ltw:add(nn.LookupTable(loader.c_vocab_size+1, opt.embedding_size))
    protos.ltw:add(nn.Dropout(opt.dropout))

    -- lti: fully connected layer + dropout for image features
    -- activations from the last fully connected layer of the deep convnet (VGG in this case)
    -- are passed through lti to get a vector of `embedding_size`
    -- linear layer dimensions are 4096 (size of fc7 layer) x `embedding_size`
    
    protos.tanh = nn.Sequential()
    protos.tanh:add(nn.Tanh())

    protos2.lti = nn.Sequential()
    protos2.lti:add(nn.Linear(4096, opt.embedding_size))
    protos2.lti:add(nn.Tanh())
    -- protos2.lti:add(nn.Dropout(opt.dropout))
    -- newest

    -- lstm: long short-term memory cell which takes a vector of size `embedding_size` at every time step
    -- hidden state h_t of LSTM cell in first layer is passed as input x_t of cell in second layer and so on.
    if opt.model == 'lstm' then
        print("we are doing LSTM")
        protos.lstm = LSTM.create(opt.embedding_size, opt.rnn_size, opt.num_layers)
    elseif opt.model == 'gru' then
        print("we are doing GRU")
        protos.lstm = GRU.create(opt.embedding_size, opt.rnn_size, opt.num_layers)
    elseif opt.model == 'rnn' then
        print("we are doing RNN")
        protos.lstm = RNN.create(opt.embedding_size, opt.rnn_size, opt.num_layers)
    end
    -- sm: linear layer + softmax over the image feature
    -- linear layer dimensions are `rnn_size` x 4096
    

    

    -- negative log-likelihood loss
    
    l1hinge = nn.L1HingeEmbeddingCriterion(1)
    l2 = nn.MSECriterion()
    l3 = nn.MSECriterion()
    protos.criterion = nn.ParallelCriterion():add(l1hinge,0.5):add(l2,0.25):add(l3,0.25)


    -- Creates a criterion that measures the loss given an input x = {x1, x2}, a table of two Tensors, 
    -- and a label y (1 or -1): this is used for measuring whether two inputs are similar or dissimilar, 
    -- using the L1 distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.


    -- pass over the model to gpu
    if opt.gpuid >= 0 then
        protos.ltw = protos.ltw:cuda()
        protos2.lti = protos2.lti:cuda()
        protos.lstm = protos.lstm:cuda()
        protos.criterion = protos.criterion:cuda()
    end
end

-- put all trainable model parameters into one flattened parameters tensor
params, grad_params = utils.combine_all_parameters(protos2.lti, protos.lstm) 

print('Parameters: ' .. params:size(1))
print('Batches: ' .. loader.batch_data.train.nbatches)

print ('train' .. loader.batch_data.train.nbatches)
print ('val' .. loader.batch_data.val.nbatches)

-- initialize model parameters
if do_random_init then
    params:uniform(-0.08, 0.08)
end

-- make clones of the LSTM model that shared parameters for subsequent timesteps (unrolling)
lstm_clones1 = {}
lstm_clones1 = utils.clone_many_times(protos.lstm, loader.c_max_length + 1)

lstm_clones2 = {}
lstm_clones2 = utils.clone_many_times(protos.lstm, loader.c_max_length + 1)

lstm_clones3 = {}
lstm_clones3 = utils.clone_many_times(protos.lstm, loader.c_max_length + 1)

lstm_clones4 = {}
lstm_clones4 = utils.clone_many_times(protos.lstm, loader.c_max_length + 1)

-- initialize h_0 and c_0 of LSTM to zero tensors and store in `init_state`
init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    h_init = h_init:cuda() 
    table.insert(init_state, h_init:clone())

    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- make a clone of `init_state` as it's going to be modified later
local init_state_global = utils.clone_list(init_state)
print('By')
-- closure to calculate accuracy over validation set
feval_val = function(max_batches)
    print("snapshots")
    count = 0
    n = loader.batch_data.val.nbatches
    print(n)
    -- set `n` to `max_batches` if provided
    if max_batches ~= nil then n = math.min(n, max_batches) end
    print(n)
    -- set to evaluation mode for dropout to work properly
    protos.ltw:evaluate()
    -- protos.lti:evaluate() newest
    print('By')
    for i = 1, n do

        -- load caption batch, tag batch and image features batch
        c1_batch, c2_batch, t_batch, imf1_batch, imf2_batch = loader:next_batch('val')

        -- 1st index of `nn.LookupTable` is reserved for zeros
        c1_batch = c1_batch + 1
        c2_batch = c2_batch + 1

        -- forward the caption features through ltw
        cf1 = protos.ltw:forward(c1_batch)
        cf2 = protos.ltw:forward(c2_batch)

        -- forward the image features through lti
        imf1 = protos2.lti:forward(imf1_batch)  
        imf2 = protos2.lti:forward(imf2_batch)

        -- convert to CudaTensor if using gpu
        if opt.gpuid >= 0 then
            imf1 = imf1:cuda()
            imf2 = imf2:cuda()

        end   

        -- set the state at 0th time step of LSTM
        rnn_state1 = {[0] = init_state_global}
        rnn_state2 = {[0] = init_state_global}
        rnn_state3 = {[0] = init_state_global}
        rnn_state4 = {[0] = init_state_global}

        -- LSTM forward pass for caption features
        left1 = torch.Tensor(1,52)
        lefts1 = left1:storage()
        left2 = torch.Tensor(1,52)
        lefts2 = left2:storage()

        -- LSTM forward pass for caption features
        for t = 1, loader.c_max_length do
            lstm_clones1[t]:training()
            lstm_clones2[t]:training()
            lstm_clones3[t]:training()
            lstm_clones4[t]:training()

            -- lst1 and lst2 is used for caption lstm
            -- lst3 and lst4 is used for caption + imf
            lst1 = lstm_clones1[t]:forward{cf1:select(2,t), unpack(rnn_state1[t-1])}
            lst2 = lstm_clones2[t]:forward{cf2:select(2,t), unpack(rnn_state2[t-1])}

            lst3 = lstm_clones3[t]:forward{lst1[#lst1]+imf1, unpack(rnn_state3[t-1])}
            lst4 = lstm_clones4[t]:forward{lst2[#lst1]+imf2, unpack(rnn_state4[t-1])}
            
            lefts1 = lefts1 + protos.tanh:forward(lst3[#lst3])
            lefts2 = lefts2 + protos.tanh:forward(lst4[#lst4])

            -- at every time step, set the rnn state (h_t, c_t) to be passed as input in next time step
            rnn_state1[t] = {}
            rnn_state2[t] = {}
            rnn_state3[t] = {}
            rnn_state4[t] = {}


            for i = 1, #init_state do table.insert(rnn_state1[t], lst1[i]) end
            for i = 1, #init_state do table.insert(rnn_state2[t], lst2[i]) end
            for i = 1, #init_state do table.insert(rnn_state3[t], lst3[i]) end
            for i = 1, #init_state do table.insert(rnn_state4[t], lst4[i]) end

            -- a = t
        end
    
        -- after completing the unrolled LSTM forward pass with caption features, forward the image features
        -- lst = lstm_clones[loader.c_max_length + 1]:forward{imf, unpack(rnn_state[loader.c_max_length])}
        -- newest

        -- forward the hidden state at the last time step to get softmax over image features(rnn_size * 4096)
        -- prediction1 = protos2.lti:forward(lst1[#lst1])
        for t = 1, #init_state do
            prediction1 = prediction1 + rnn_state1[t]
        end

        prediction2 = protos2.lti:forward(imf)


        if opt.gpuid >= 0 then
            prediction1 = prediction1:cuda()
            prediction2 = prediction2:cuda()

        end   
    
        
        -- -- count number of correct answers
        -- 换criterion要改
        -- _, idx  = prediction:max(2)
        -- for j = 1, opt.batch_size do
        --     if idx[j][1] == t_batch[j] then
        --         count = count + 1
        --     end
        -- end

    end

    -- set to training mode once done with validation
    protos.ltw:training()
    -- protos.lti:training() 

    -- return accuracy
    return count / (n * opt.batch_size)

end

-- closure to run a forward and backward pass and return loss and gradient parameters
feval = function(x)
    -- get latest parameters
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    -- load caption batch, tag batch and image features batch
    c1_batch, c2_batch, t_batch, imf1_batch, imf2_batch = loader:next_batch()

    -- slightly hackish; 1st index of `nn.LookupTable` is reserved for zeros
    c1_batch = c1_batch + 1
    c2_batch = c2_batch + 1
    

    -- forward the caption features through ltw
    cf1 = protos.ltw:forward(c1_batch)
    cf2 = protos.ltw:forward(c2_batch)


    -- forward the image features through lti
    imf1 = protos2.lti:forward(imf1_batch)  
    imf2 = protos2.lti:forward(imf2_batch)
    
    -- convert to CudaTensor if using gpu
    if opt.gpuid >= 0 then
        imf1 = imf1:cuda()
        imf2 = imf2:cuda()
        -- t_batch = t_batch:cuda()
    end   


    -- 以上，过
    ------------ forward pass ------------

    -- set initial loss
    loss = 0

    -- set the state at 0th time step of LSTM
    rnn_state1 = {[0] = init_state_global}
    rnn_state2 = {[0] = init_state_global}
    rnn_state3 = {[0] = init_state_global}
    rnn_state4 = {[0] = init_state_global}

    left1 = torch.Tensor(1,52)
    lefts1 = left1:storage()
    left2 = torch.Tensor(1,52)
    lefts2 = left2:storage()

    if opt.gpuid >= 0 then
        left1 = left1:cuda()
        left2 = left2:cuda()
        -- t_batch = t_batch:cuda()
    end   

    -- LSTM forward pass for caption features
    for t = 1, loader.c_max_length do
        lstm_clones1[t]:training()
        lstm_clones2[t]:training()
        lstm_clones3[t]:training()
        lstm_clones4[t]:training()

        -- lst1 and lst2 is used for caption lstm
        -- lst3 and lst4 is used for caption + imf
        -- print (cf1:select(2,t):size())
        lst1 = lstm_clones1[t]:forward{cf1:select(2,t), unpack(rnn_state1[t-1])}
        lst2 = lstm_clones2[t]:forward{cf2:select(2,t), unpack(rnn_state2[t-1])}

        -- print (lst1[#lst1]:size())
        -- print (imf1:size())
        lst3 = lstm_clones3[t]:forward{lst1[#lst1]+imf1, unpack(rnn_state3[t-1])}
        lst4 = lstm_clones4[t]:forward{lst2[#lst1]+imf2, unpack(rnn_state4[t-1])}
        
        -- aa = protos2.lti:forward(lst3[#lst3])
        -- print (aa:size())



        -- left1 = left1 + protos.tanh:forward(lst3[#lst3])
        -- left2 = left2 + protos.tanh:forward(lst4[#lst4])

        -- at every time step, set the rnn state (h_t, c_t) to be passed as input in next time step
        rnn_state1[t] = {}
        rnn_state2[t] = {}
        rnn_state3[t] = {}
        rnn_state4[t] = {}


        for i = 1, #init_state do table.insert(rnn_state1[t], lst1[i]) end
        for i = 1, #init_state do table.insert(rnn_state2[t], lst2[i]) end
        for i = 1, #init_state do table.insert(rnn_state3[t], lst3[i]) end
        for i = 1, #init_state do table.insert(rnn_state4[t], lst4[i]) end

        -- a = t
    end

    -- after completing the unrolled LSTM forward pass with caption features, forward the image features
    -- lst = lstm_clones[loader.c_max_length + 1]:forward{imf, unpack(rnn_state[loader.c_max_length])}
    -- 1
    -- lst = lstm_clones[loader.c_max_length + 1]:forward{imf, unpack(rnn_state[loader.c_max_length])}
    -- newest

    -- forward the hidden state at the last time step to get softmax over image features(answer)
    -- prediction1 = lst1[#lst1]
    -- prediction1 = protos2.lti:forward(lst1[#lst1])
    -- prediction2 = lst2[#lst2]
    for t = 1, #init_state do
        prediction1 = prediction1 + rnn_state1[t]
        left1 = left1[t]
        left2 = left2[t]
    end
    prediction2 = protos2.lti:forward(imf)
    
    sp1 = prediction1:storage()
    sp2 = prediction2:storage()


    for i=1, sp1:size() do
        if sp1[i] <= 0 then
            sp1[i] = 0
        else
            sp1[i] = 1
        end
    end

    for i=1, sp2:size() do
        if sp2[i] <= 0 then
            sp2[i] = 0
        else
            sp2[i] = 1
        end
    end



    sf1 = imf1:storage()
    sf2 = imf2:storage()

    
    for i=1, sf1:size() do
        if sf1[i] <= 0 then
            sf1[i] = 0
        else
            sf1[i] = 1
        end
    end

    for i=1, sf2:size() do
        if sf2[i] <= 0 then
            sf2[i] = 0
        else
            sf2[i] = 1
        end
    end

    if opt.gpuid >= 0 then
        lefts1 = lefts1:cuda()
        lefts2 = lefts2:cuda()

    end


    -- calculate loss
    -- 换criterion要改

    -- protos.criterion = 
    -- protos.criterion2 = 
    -- protos.criterion3 = nn.MSECriterion()

    inputs = {{lefts1,lefts2}, prediction1, prediction2}
    outputs = {t_batch,lefts1,lefts2}
    loss = protos.criterion:forward(inputs, outputs)
    -- loss2 = protos.criterion2:forward()
    -- loss3 = protos.criterion3:forward()


    ------------ backward pass ------------

    -- backprop through loss and softmax
    -- 换criterion要改
    dloss = protos.criterion:backward(inputs, outputs)
    
    -- doutput_t = protos.sm:backward(lst[#lst], dloss[1])  -- <<<<<<
    
    if opt.model == 'lstm' then
        num = 2
    else
        num = 1
    end
    -- set internal state of LSTM (starting from last time step)
    drnn_state1 = {[loader.c_max_length ] = utils.clone_list(init_state, true)}
    drnn_state1[loader.c_max_length][opt.num_layers * num] = dloss[1]

    drnn_state2 = {[loader.c_max_length ] = utils.clone_list(init_state, true)}
    drnn_state2[loader.c_max_length][opt.num_layers * num] = dloss[1]

    drnn_state3 = {[loader.c_max_length ] = utils.clone_list(init_state, true)}
    drnn_state3[loader.c_max_length][opt.num_layers * num] = dloss[1]

    drnn_state4 = {[loader.c_max_length ] = utils.clone_list(init_state, true)}
    drnn_state4[loader.c_max_length][opt.num_layers * num] = dloss[1]
   
    dcf1 = torch.Tensor(cf1:size()):zero()
    dcf2 = torch.Tensor(cf2:size()):zero()

    if opt.gpuid >= 0 then
        dcf1 = dcf1:cuda()
        dcf2 = dcf2:cuda()
        
    end
    
   
    -- backprop into the LSTM for rest of the time steps
    for t = loader.c_max_length, 1, -1 do
        dlst1 = lstm_clones1[t]:backward({cf1:select(2, t), unpack(rnn_state1[t-1])}, drnn_state1[t])
        dcf1:select(2, t):copy(dlst1[1])
        drnn_state1[t-1] = {}

        dlst2 = lstm_clones2[t]:backward({cf2:select(2, t), unpack(rnn_state2[t-1])}, drnn_state2[t])
        dcf2:select(2, t):copy(dlst2[1])
        drnn_state2[t-1] = {}

        dlst3 = lstm_clones3[t]:backward({lst1[#lst1]+imf1, unpack(rnn_state3[t-1])}, drnn_state3[t])
        dcf3:select(2, t):copy(dlst3[1])
        drnn_state3[t-1] = {}

        dlst4 = lstm_clones4[t]:backward({lst1[#lst2]+imf2, unpack(rnn_state4[t-1])}, drnn_state4[t])
        dcf4:select(2, t):copy(dlst4[1])
        drnn_state4[t-1] = {}

        for i,v in pairs(dlst) do
            if i > 1 then
                drnn_state[t-1][i-1] = v
            end
        end
    end
    -- print(dqf)
    -- zero gradient buffers of lookup table, backprop into it and update parameters
    protos.ltw:zeroGradParameters()
    protos.ltw:backward(c1_batch, dcf1)
    protos.ltw:backward(c2_batch, dcf2)
    protos.ltw:updateParameters(opt.learning_rate)
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)


    return loss, grad_params

end

-- optim state with ADAM parameters
local optim_state = {learningRate = opt.learning_rate, alpha = opt.alpha, beta = opt.beta, epsilon = opt.epsilon}

-- train / val loop!
losses = {}
iterations = opt.max_epochs * loader.batch_data.train.nbatches
print('Max iterations: ' .. iterations)
lloss = 0

for i = 1, iterations do

    _, local_loss = optim.adam(feval, params, optim_state)

    losses[#losses + 1] = local_loss[1]

    lloss = lloss + local_loss[1]
    local epoch = i / loader.batch_data.train.nbatches

    if i%10 == 0 then
        print('epoch ' .. epoch .. ' loss ' .. lloss / 10)
        lloss = 0
    end

    -- Decay learning rate occasionally
    if i % loader.batch_data.train.nbatches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- Calculate validation accuracy and save model snapshot
    if i % opt.save_every == 0 or i == iterations then
        print('Checkpointing. Calculating validation accuracy..')
        local val_acc = feval_val()
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_acc)
        print('Saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.opt = opt
        checkpoint.protos = protos
        checkpoint.vocab_size = loader.c_vocab_size
        checkpoint.imf = imf
        torch.save(savefile, checkpoint)
    end

    if i%10 == 0 then
        collectgarbage()
    end
end

