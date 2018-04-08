
local DataLoader = {}
DataLoader.__index = DataLoader


function DataLoader.create(data_dir, batch_size, opt, mode)

    local self = {}
    setmetatable(self, DataLoader)

    self.mode = mode or 'train'

    local train_captions_file = path.join(data_dir, 'image_train_info.json')
   
    local val_captions_file = path.join(data_dir, 'image_val_info.json')

    local captions_vocab_file =  path.join(data_dir, 'captions_vocab.t7')

    local tensor_file =  path.join(data_dir, 'data.t7')

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
        ['caption1'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.c_max_length),
        ['image_feat1'] = torch.DoubleTensor(self.batch_size * math.floor(#data.train / self.batch_size), 4096),
        ['image_id1'] = {},
        ['caption2'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.c_max_length),
        ['image_feat2'] = torch.DoubleTensor(self.batch_size * math.floor(#data.train / self.batch_size), 4096),
        ['image_id2'] = {},
        ['tag'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size)), --结果是列
        ['nbatches'] = math.floor(#data.train / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.train.image_feat = self.batch_data.train.image_feat:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.train.nbatches do
        self.batch_data.train.caption[i] = data.train[i]['caption1']
        if not data.train[i]['tag'] then
            data.train[i]['tag'] = 0
        end
        self.batch_data.train.tag[i] = data.train[i]['tag']
        if not fc7[fc7_mapping[data.train[i]['image_id']]] then
            self.batch_data.train.image_feat[i] = 0
        else
            self.batch_data.train.image_feat[i] = fc7[fc7_mapping[data.train[i]['image_id']]]
        end
        self.batch_data.train.image_id[i] = data.train[i]['image_id']
    end

    if opt.gpuid >= 0 then
        self.batch_data.train.caption = self.batch_data.train.caption:cuda()
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
        ['caption'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size), data.c_max_length),
        ['tag'] = torch.ShortTensor(self.batch_size * math.floor(#data.val / self.batch_size)),
        ['image_feat'] = torch.DoubleTensor(self.batch_size * math.floor(#data.val / self.batch_size), 4096),
        ['image_id'] = {},
        ['nbatches'] = math.floor(#data.val / self.batch_size)
    }

    if opt.gpuid >= 0 then
        self.batch_data.val.image_feat = self.batch_data.val.image_feat:cuda()
    end

    for i = 1, self.batch_size * self.batch_data.val.nbatches do
        self.batch_data.val.caption[i] = data.val[i]['caption']
        if not data.val[i]['tag'] then
            data.val[i]['tag'] = 0
        end
        self.batch_data.val.tag[i] = data.val[i]['tag']
        if not fc7[fc7_mapping[data.val[i]['image_id']]] then
            self.batch_data.val.image_feat[i] = 0
        else
            self.batch_data.val.image_feat[i] = fc7[fc7_mapping[data.val[i]['image_id']]]
        end
        self.batch_data.val.image_id[i] = data.val[i]['image_id']
    end

    if opt.gpuid >= 0 then
        self.batch_data.val.caption = self.batch_data.val.caption:cuda()
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
        local caption = self.batch_data.train.caption:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local tag = self.batch_data.train.tag:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image = self.batch_data.train.image_feat:narrow(1, (self.train_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id = {unpack(self.batch_data.train.image_id, (self.train_batch_idx - 1) * self.batch_size + 1, self.train_batch_idx * self.batch_size)}

        self.train_batch_idx = self.train_batch_idx + 1

        return caption, tag, image, image_id
    else
        if self.val_batch_idx - 1 == self.batch_data.val.nbatches then self.val_batch_idx = 1 end
        local caption = self.batch_data.val.caption:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local tag = self.batch_data.val.tag:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image = self.batch_data.val.image_feat:narrow(1, (self.val_batch_idx - 1) * self.batch_size + 1, self.batch_size)
        local image_id = {unpack(self.batch_data.val.image_id, (self.val_batch_idx - 1) * self.batch_size + 1, self.val_batch_idx * self.batch_size)}

        self.val_batch_idx = self.val_batch_idx + 1

        print ('222')
        return caption, tag, image, image_id
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


    for i=1,82080 do
        local count = 0
        for token in word_iter(train_c[tostring(i)]['captions']) do
            if not unordered[token] then
                unordered[token] = 1
            else
                unordered[token] = unordered[token] + 1
            end
            count = count + 1
        end
        if count > max_length then max_length = count end

    end

    for i=1, 40503 do
        local count = 0
        if (type(val_c[tostring(i)]['captions'])=='string') then
            for token in word_iter(val_c[tostring(i)]['captions']) do
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

    for i=1, 82080 do
        local caption = {}
        local wl = 0
        for token in word_iter(train_c[tostring(i)]['captions']) do
            wl = wl + 1
            caption[wl] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
        end
        data.train[idx] = {
            image_id = train_c[tostring(i)]['image_id'],
            caption = torch.ShortTensor(max_length):zero(),
            tag = train_c[tostring(i)]['image_tag'] == 'yes' and 1 or -1
        }
        for j = 1, wl do
            data.train[idx]['caption'][max_length - wl + j] = caption[j]
        end
        idx = idx + 1

    end

    idx = 1

    for i=1, 40503 do
        local caption = {}
        local wl = 0
        if (type(val_c[tostring(i)]['captions'])=='string') then
            for token in word_iter(val_c[tostring(i)]['captions']) do
                wl = wl + 1
                caption[wl] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
            end
        
            data.val[idx] = {
                image_id = val_c[tostring(i)]['image_id'],
                caption = torch.ShortTensor(max_length):zero(),
                tag = val_c[tostring(i)]['image_tag'] == 'yes' and 1 or -1
            }
            for j = 1, wl do
                data.val[idx]['caption'][max_length - wl + j] = caption[j]
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

