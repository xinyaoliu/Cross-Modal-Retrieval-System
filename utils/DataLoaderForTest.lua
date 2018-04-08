
local DataLoaderForTest = {}
DataLoaderForTest.__index = DataLoaderForTest


function DataLoaderForTest.create(data_dir, batch_size, opt, mode)

    local self = {}
    setmetatable(self, DataLoaderForTest)

 

    local tensor_file =  path.join(data_dir, 'testdata.t7')

   

    print('Loading data files...')
    local data = torch.load(tensor_file)
    if mode == 'fc7_feat' then
        self.data = data
        collectgarbage()
        return self
    end

 

    if mode == 'predict' then
        -- self.data = data
        collectgarbage()
        return self
    end

    self.test_nbatches = 0


    print('Loading train fc7 features from /home/liuxinyao/bs/code/test_fc7.t7' )
    local fc7 = torch.load('/home/liuxinyao/bs/code/test_fc7.t7')
    local fc7_image_id = torch.load('/home/liuxinyao/bs/code/test_fc7_image_id.t7')
    print ('Loading done')

    local fc7_mapping = {}
    for i, v in pairs(fc7_image_id) do
        fc7_mapping[v] = i
    end
   

    self.batch_data = {['test'] = {}}

    self.batch_data.test = {
    --     ['caption'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size), data.c_max_length),
    --     ['tag'] = torch.ShortTensor(self.batch_size * math.floor(#data.train / self.batch_size)), --结果是列
        ['image_feat'] = torch.DoubleTensor(4000 * math.floor(#data.test / 4000), 4096),
        ['image_id'] = {},
        ['nbatches'] = math.floor(#data.test / 4000)
    }

    if opt.gpuid >= 0 then
        self.batch_data.test.image_feat = self.batch_data.test.image_feat:cuda()
    end

    for i = 1, 4000 * self.batch_data.test.nbatches do
        if not fc7[fc7_mapping[data.test[i]['image_id']]] then
            self.batch_data.test.image_feat[i] = 0
        else
            self.batch_data.test.image_feat[i] = fc7[fc7_mapping[data.test[i]['image_id']]]
        end
        self.batch_data.test.image_id[i] = data.test[i]['image_id']
    end

    self.test_batch_idx = 1

    print ('used to come here')

    collectgarbage()
    return self

end





function DataLoaderForTest:next_batch_for_test()

        if self.test_batch_idx - 1 == self.batch_data.test.nbatches then self.test_batch_idx = 1 end
        local image = self.batch_data.test.image_feat:narrow(1, (self.test_batch_idx - 1) * 4000 + 1, 4000)
        local image_id = {unpack(self.batch_data.test.image_id, (self.test_batch_idx - 1) * 4000 + 1, self.test_batch_idx * 4000)}

        self.test_batch_idx = self.test_batch_idx + 1

        return image, image_id
   
        
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

return DataLoaderForTest

