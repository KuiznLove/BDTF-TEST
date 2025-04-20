import pickle

file_path = "../../V2/14lap/vocab_deprel.vocab"

with open(file_path, "rb") as f:  # 注意用二进制模式 "rb"
    obj = pickle.load(f)

# 打印内容（假设是词汇表对象）
print("索引到标签 (itos):", obj.itos)  # 类似 ["<pad>", "<unk>", "punct", ...]
print("标签到索引 (stoi):", obj.stoi)  # 类似 {"<pad>": 0, "<unk>": 1, ...}