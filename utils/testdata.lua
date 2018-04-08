-- function DataLoader.json_to_tensor(in_train_c, in_val_c, out_vocab_c, out_tensor)

out_tensor = '/home/liuxinyao/bs/code/testdata.t7'

local JSON = (loadfile "JSON.lua")()

print('creating vocabulary mapping...')



-- build caption vocab using train+val

f = torch.DiskFile('/home/liuxinyao/bs/code/image_info_test.json')
c = f:readString('*a')
local test_c = JSON:decode(c)
f:close()



-- unordered = {}
max_length = 0


local threshold = 0
-- local ordered = {}
-- for token, count in pairs(unordered) do
--     if count > threshold then
--         ordered[#ordered + 1] = token
--     end
-- end
-- ordered[#ordered + 1] = "UNK"
-- table.sort(ordered)

-- local c_vocab_mapping = {}
-- for i, word in ipairs(ordered) do
--     c_vocab_mapping[word] = i
-- end

print('putting data into tensor...')

-- save train+val data

local data = {
    test = {},
}


local idx = 1

for i=1, 40775 do
    local wl = 0
    data.test[idx] = {
        image_id = test_c[tostring(i)]['image_id'],

    }
    -- for j = 1, wl do
    --     data.train[idx]['caption'][max_length - wl + j] = caption[j]
    -- end
    idx = idx + 1

end

print (data)

-- idx = 1

-- for i=1, 40775 do
--     local caption = {}
--     local wl = 0
--     if (type(val_c[tostring(i)]['captions'])=='string') then
--         for token in word_iter(val_c[tostring(i)]['captions']) do
--             wl = wl + 1
--             caption[wl] = c_vocab_mapping[token] or c_vocab_mapping["UNK"]
--         end
    
--         data.val[idx] = {
--             image_id = val_c[tostring(i)]['image_id'],
--             caption = torch.ShortTensor(max_length):zero(),
--             tag = val_c[tostring(i)]['image_tag'] == 'yes' and 1 or -1
--         }
--         for j = 1, wl do
--             data.val[idx]['caption'][max_length - wl + j] = caption[j]
--         end
--         idx = idx + 1
--     end
-- end


-- save output preprocessed files



print('saving ' .. out_tensor)
torch.save(out_tensor, data)

-- end