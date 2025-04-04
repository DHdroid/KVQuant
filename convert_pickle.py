pickle_path = "/home/ubuntu/donghyun/vistool/data/quantizers_meta-llama_Llama-2-7b-chat-hf.pickle"
output_path = "/home/ubuntu/donghyun/vistool/data/quantizers_meta-llama_Llama-2-7b-chat-hf_torch.pickle"

import pickle
import torch
with open(pickle_path, "rb") as f:
    q = pickle.load(f)

for key in q.keys():
    q[key][-1][0] = torch.from_numpy(q[key][-1][0])

with open(output_path, "wb") as f:
    pickle.dump(q, f)
