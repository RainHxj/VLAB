import pickle
import ipdb

data_path = "/PATH/datasets/MSRVTT/msrvtt/MSRVTT_Caption.pkl"
save_path = "msrvtt_val_caption.pkl"
with open(data_path,"rb") as f:
    data_info = pickle.load(f)

new_data = []
for info in data_info:
    # ipdb.set_trace()
    v = {}
    d = info
    filename=d["filename"]
    captions = eval(d['caption'])
    text = []
    for caption in captions:
        text.append(caption['caption'])

    # ipdb.set_trace()
    v["filename"] = filename
    v["text"]=text
    new_data.append(v)

with open(save_path, "wb") as fw:
    pickle.dump(new_data, fw)