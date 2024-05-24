import pickle
import random

# msvd_path = "/PATH/datasets/ds_datasets/msvd/train_anno_new.pkl"
# save_path="msvd_train_caption_split.pkl"

# anns = pickle.load(open(msvd_path, "rb"))

# new_data=[]

# for ann in anns:
    
#     filename=ann["filename"]
#     texts = ann["text"]
#     for text in texts:
#         info={}
#         info["filename"]=filename
#         info["text"]=text
#         new_data.append(info)

# random.shuffle(new_data)


# with open(save_path, "wb") as fw:
#     pickle.dump(new_data, fw)


save_path = "msvd_test_caption_split.pkl"

test_path = "/PATH/datasets/ds_datasets/msvd/test_anno_new.pkl"

anns = pickle.load(open(test_path, "rb"))

new_data=[]

for ann in anns:
    info = {}
    filename=ann["filename"]
    text = ""

    info["filename"]=filename
    info['text']=text

    new_data.append(info)



with open(save_path, "wb") as fw:
    pickle.dump(new_data, fw)