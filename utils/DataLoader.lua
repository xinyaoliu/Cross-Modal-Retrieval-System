
local DataLoader = {}
DataLoader.__index = DataLoader


function DataLoader.create(data_dir, batch_size, opt, mode)

    local self = {}
    setmetatable(self, DataLoader)

    self.mode = mode or 'train'

    local train_captions_file = path.join(data_dir, 'image_train_info_rectified_small.json')
   
    local val_captions_file = path.join(data_dir, 'image_val_info_rectified_small.json')

    local captions_vocab_file =  path.join(data_dir, 'captions_vocab.t7')

    local tensor_file =  path.join(data_dir, 'data_rectified.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing

    local run_prepro = false
    if not (path.exists(captions_vocab_file) and path.exists(tensor_file)) then
        print('captions_vocab.t7 or data.t7 files do not exist. Running preprocessing...')
        run_prepro = true
    else
        local captions_vocab_attr = lfs.attributes(captions_vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)

        if captions_vocab_attr.modification > tensor_attr.modification then
            print('captions_vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing...')
        DataLoader.json_to_tensor(train_captions_file, val_captions_file, captions_vocab_file, tensor_file)
    end

    print('Loading data files...')
    local data = torch.load(tensor_file)
    if mode == 'fc7_feat' then
        self.data = data
        collectgarbage()
        return self
    end

    self.c_max_length = data.c_max_length
    self.c_vocab_mapping = torch.load(captions_vocab_file)

    self.c_vocab_size = 0
    for _ in pairs(self.c_vocab_mapping) do
        self.c_vocab_size = self.c_vocab_size + 1
    end

    
    self.batch_size = batch_size

    if mode == 'predict' then
        -- self.data = data
        collectgarbage()
        return self
    end

    self.train_nbatches = 0
    self.val_nbatches = 0

    -- Load train into batches

    print('Loading train fc7 features from /home/liuxinyao/bs/code/data/train_fc7.t7' )
    local fc7 = torch.load('/home/liuxinyao/bs/code/data/train_fc7.t7')
    local fc7_image_id = torch.load('/home/liuxinyao/bs/code/data/train_fc7_image_id.t7')
    print ('Loading done')
    local fc7_mapping = {}
    for i, v in pairs(fc7_image_id) do
        fc7_mapping[v] = i
    end



    self.batch_data = {['train'] = {}, ['val'] = {}}

    self.batch_data.train = {
        ['image_id1'] = {},
        ['image_id2'] = {},
        ['caption1'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.c_max_length),
        ['caption2'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.c_max_length),
        ['tag'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size)), --结果是列
        ['image_feat1'] = torch.DoubleTensor(self.batch_size * math.floor(#data.train / self.batch_size), 4096),
        ['image_feat2'] = torch.DoubleTensor(self.batch_size * math.floor(#data.train / self.batch_size), 4096),
        ['nbatches'] = math.floor(#data.train / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.train.image_feat1 = self.batch_data.train.image_feat1:cuda()
        self.batch_data.train.image_feat2 = self.batch_data.train.image_feat2:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.train.nbatches do
        self.batch_data.train.caption1[i] = data.train[i]['caption1']
        self.batch_data.train.caption2[i] = data.train[i]['caption2']
        -- if not data.train[i]['tag'] then
        --     data.train[i]['tag'] = 0
        -- end 
        -- ???
        self.batch_data.train.tag[i] = data.train[i]['tag']
        if not fc7[fc7_mapping[data.train[i]['image_id1']]] then
            self.batch_data.train.image_feat1[i] = 0
        else
            self.batch_data.train.image_feat1[i] = fc7[fc7_mapping[data.train[i]['image_id1']]]
        end

        if not fc7[fc7_mapping[data.train[i]['image_id2']]] then
            self.batch_data.train.image_feat2[i] = 0
        else
            self.batch_data.train.image_feat2[i] = fc7[fc7_mapping[data.train[i]['image_id2']]]
        end

        self.batch_data.train.image_id1[i] = data.train[i]['image_id1']
        self.batch_data.train.image_id2[i] = data.train[i]['image_id2']
    end

    if opt.gpuid >= 0 then
        self.batch_data.train.caption1 = self.batch_data.train.caption1:cuda()
        self.batch_data.train.caption2 = self.batch_data.train.caption2:cuda()
        self.batch_data.train.tag = self.batch_data.train.tag:cuda()
    end

    -- Load val into batches

    print('Loading val fc7 features from /home/liuxinyao/bs/code/data/val_fc7.t7')
    local fc7 = torch.load('/home/liuxinyao/bs/code/data/val_fc7.t7')
    local fc7_image_id = torch.load('/home/liuxinyao/bs/code/data/val_fc7_image_id.t7')
    local fc7_mapping = {}
    for i, v in pairs(fc7_image_id) do
        fc7_mapping[v] = i
    end

    self.batch_data.val = {
        ['image_id1'] = {},
        ['image_id2'] = {},
        ['caption1'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size), data.c_max_length),
        ['caption2'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size), data.c_max_length),
        ['tag'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size)),
        ['image_feat1'] = torch.DoubleTensor(self.batch_size * math.floor(#data.val / self.batch_size), 4096),
        ['image_feat2'] = torch.DoubleTensor(self.batch_size * math.floor(#data.val / self.batch_size), 4096),
        ['nbatches'] = math.floor(#data.val / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.val.image_feat1 = self.batch_data.val.image_feat1:cuda()
        self.batch_data.val.image_feat2 = self.batch_data.val.image_feat2:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.val.nbatches do
        self.batch_data.val.caption1[i] = data.val[i]['caption1']
        self.batch_data.val.caption2[i] = data.val[i]['caption2']
        if not data.val[i]['tag'] then
            data.val[i]['tag'] = 0
        end
        self.batch_data.val.tag[i] = data.val[i]['tag']
        if not fc7[fc7_mapping[data.val[i]['image_id1']]] then
            self.batch_data.val.image_feat1[i] = 0
        else
            self.batch_data.val.image_feat1[i] = fc7[fc7_mapping[data.val[i]['image_id1']]]
        end

        if not fc7[fc7_mapping[data.val[i]['image_id2']]] then
            self.batch_data.val.image_feat2[i] = 0
        else
            self.batch_data.val.image_feat2[i] = fc7[fc7_mapping[data.val[i]['image_id2']]]
        end

        self.batch_data.val.image_id1[i] = data.val[i]['image_id1']
        self.batch_data.val.image_id2[i] = data.val[i]['image_id2']
    end

    if opt.gpuid >= 0 then
        self.batch_data.val.caption1 = self.batch_data.val.caption1:cuda()
        self.batch_data.val.caption2 = self.batch_data.val.caption2:cuda()
        self.batch_data.val.tag = self.batch_data.val.tag:cuda()
    end

    self.train_batch_idx = 1
    self.val_batch_idx = 1

    collectgarbage()
    return self

end


-- self.fc7_mapping = fc7_mapping
-- self.fc7 = fc7
-- self.test = 3

function DataLoader:next_batch(split)
    split = split or 'train'
    if split == 'train' then
        if self.train_batch_idx - 1 == self.batch_data.train.nbatches then self.train_batch_idx = 1 end
        local caption1 = self.batch_data.train.caption1:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local caption2 = self.batch_data.train.caption2:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local tag = self.batch_data.train.tag:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image1 = self.batch_data.train.image_feat1:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image2 = self.batch_data.train.image_feat2:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id1 = {unpack(self.batch_data.train.image_id1, (self.train_batch_idx - 1) * self.batch_size + 1, self.train_batch_idx * self.batch_size)}
        local image_id2 = {unpack(self.batch_data.train.image_id2, (self.train_batch_idx - 1) * self.batch_size + 1, self.train_batch_idx * self.batch_size)}

        self.train_batch_idx = self.train_batch_idx + 1

        return caption1,caption2, tag, image1, image2, image_id1, image_id2
    else
        if self.val_batch_idx - 1 == self.batch_data.val.nbatches then self.val_batch_idx = 1 end
        local caption1 = self.batch_data.val.caption1:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local caption2 = self.batch_data.val.caption2:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local tag = self.batch_data.val.tag:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image1 = self.batch_data.val.image_feat1:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image2 = self.batch_data.val.image_feat2:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id1 = {unpack(self.batch_data.val.image_id1, (self.val_batch_idx - 1) * self.batch_size + 1, self.val_batch_idx * self.batch_size)}
        local image_id2 = {unpack(self.batch_data.val.image_id2, (self.val_batch_idx - 1) * self.batch_size + 1, self.val_batch_idx * self.batch_size)}

        self.val_batch_idx = self.val_batch_idx + 1

        return caption1,caption2, tag, image1, image2, image_id1, image_id2
    end
end




function DataLoader.json_to_tensor(in_train_c, in_val_c, out_vocab_c, out_tensor)

    local JSON = (loadfile "utils/JSON.lua")()

    print('creating vocabulary mapping...')



    -- build caption vocab using train+val

    f = torch.DiskFile(in_train_c)
    c = f:readString('*a')
    local train_c = JSON:decode(c)
    f:close()

    f = torch.DiskFile(in_val_c)
    c = f:readString('*a')
    local val_c = JSON:decode(c)
    f:close()

    unordered = {}
    max_length = 0


    for i=1,29999 do
        local count = 0
        if (type(train_c[tostring(i)]['captions1']) == 'string') then
            for token in word_iter(train_c[tostring(i)]['captions1']) do
                if not unordered[token] then
                    unordered[token] = 1
                else
                    unordered[token] = unordered[token] + 1
                end
                count = count + 1
            end
        end
        if count > max_length then max_length = count end

        local count = 0
        if (type(train_c[tostring(i)]['captions2'])=='string') then
            for token in word_iter(train_c[tostring(i)]['captions2']) do
                if not unordered[token] then
                    unordered[token] = 1
                else
                    unordered[token] = unordered[token] + 1
                end
                count = count + 1
            end
        end
        if count > max_length then max_length = count end

    end

    for i=1, 9999 do
        local count = 0
        if (type(val_c[tostring(i)]['captions1'])=='string') then
            for token in word_iter(val_c[tostring(i)]['captions1']) do
                if not unordered[token] then
                    unordered[token] = 1
                else
                    unordered[token] = unordered[token] + 1
                end
                count = count + 1
            end
            if count > max_length then max_length = count end
        end

        local count = 0
        if (type(val_c[tostring(i)]['captions2'])=='string') then
            for token in word_iter(val_c[tostring(i)]['captions2']) do
                if not unordered[token] then
                    unordered[token] = 1
                else
                    unordered[token] = unordered[token] + 1
                end
                count = count + 1
            end
            if count > max_length then max_length = count end
        end
    end

    local threshold = 0
    local ordered = {}
    for token, count in pairs(unordered) do
        if count > threshold then
            ordered[#ordered + 1] = token
        end
    end
    ordered[#ordered + 1] = "UNK"
    table.sort(ordered)

    local c_vocab_mapping = {}
    for i, word in ipairs(ordered) do
        c_vocab_mapping[word] = i
    end

    print('putting data into tensor...')

    -- save train+val data

    local data = {
        train = {},
        val = {},
        c_max_length = max_length
    }

    print('c max length: ' .. max_length)

    local idx = 1

    for i=1, 29999 do
        local caption1 = {}
        local caption2 = {}
        local wl1 = 0
        local wl2 = 0
        for token in word_iter(train_c[tostring(i)]['captions1']) do
            wl1 = wl1 + 1
            caption1[wl1] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
        end

        for token in word_iter(train_c[tostring(i)]['captions2']) do
            wl2 = wl2 + 1
            caption2[wl2] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
        end

        data.train[idx] = {
            image_id1 = train_c[tostring(i)]['image_id1'],
            image_id2 = train_c[tostring(i)]['image_id2'],
            caption1 = torch.ShortTensor(max_length):zero(),
            caption2 = torch.ShortTensor(max_length):zero(),
            tag = train_c[tostring(i)]['image_tag'] 
        }
        for j = 1, wl1 do
            data.train[idx]['caption1'][max_length - wl1 + j] = caption1[j]
            
        end

        for j = 1, wl2 do
            data.train[idx]['caption2'][max_length - wl2 + j] = caption2[j]
        end

        idx = idx + 1

    end

    idx = 1

    for i=1, 9999 do
        local caption1 = {}
        local caption2 = {}

        local wl1 = 0
        local wl2 = 0

        if (type(val_c[tostring(i)]['captions1'])=='string') and (type(val_c[tostring(i)]['captions2'])=='string') then
            for token in word_iter(val_c[tostring(i)]['captions1']) do
                wl1 = wl1 + 1
                caption1[wl1] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
            end
            
            for token in word_iter(val_c[tostring(i)]['captions2']) do
                wl2 = wl2 + 1
                caption2[wl2] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
            end

            data.val[idx] = {
                image_id1 = val_c[tostring(i)]['image_id1'],
                image_id2 = val_c[tostring(i)]['image_id2'],
                caption1 = torch.ShortTensor(max_length):zero(),
                caption2 = torch.ShortTensor(max_length):zero(),
                tag = val_c[tostring(i)]['image_tag']
            }
            for j = 1, wl1 do
                data.val[idx]['caption1'][max_length - wl1 + j] = caption1[j]
               
            end

            for j = 1, wl2 do

                data.val[idx]['caption2'][max_length - wl2 + j] = caption2[j]
            end

            idx = idx + 1
        end
    end


    -- save output preprocessed files



    print('saving ' .. out_vocab_c)
    torch.save(out_vocab_c, c_vocab_mapping)
    print('saving ' .. out_tensor)
    torch.save(out_tensor, data)

end

function word_iter(str)
    return string.gmatch(str, "%a+")
end

function get_keys_sorted_by_value(tbl, sort_fn)
    local keys = {}
    for key in pairs(tbl) do
        table.insert(keys, key)
    end

    table.sort(keys, function(a, b)
        return sort_fn(tbl[a], tbl[b])
    end)

    return keys
end

return DataLoader

