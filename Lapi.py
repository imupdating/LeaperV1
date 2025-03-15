# 这个文件实现了调用模型的接口
from Lmodel import LeaperV1
import torch, Config, Ltokenizer
import os
import torch.nn.functional as F
import torch

name = "cuda" if torch.cuda.is_available() else "cpu"
print('Model on', name)
device = torch.device(name)

word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")
model = LeaperV1(
        max_len=Config.max_len,
        vocab_size=len(word2id),
        n_head=Config.n_head,
        layer_num=Config.layer_num,
        d_model=Config.d_model,
        dim_feedforward=Config.feedforward_dim
    ).to(device)

def load_model(file):
    print('Loeded model from', file)
    model.load_state_dict(torch.load(file))
    model.eval()
    return model

def wordTotokens(sentence):
    return [word2id.get(word, word2id['<UNK>']) for word in sentence]

def generate_text_wordByword(model, start_token, max_len=Config.max_len, temperature=0.1, print_=False):
    model.eval()
    with torch.no_grad():
        if print_:
            for c in start_token:
                print(id2word[c], end='', flush=True) 
        
        input_seq = torch.tensor(start_token, 
                               device=device).unsqueeze(0)
        
        for _ in range(max_len-len(start_token)):
            seq_len = input_seq.size(1)
            # 注意：这里仍然要生成掩码
            mask = model.generate_mask(seq_len)
            
            output = model(input_seq, src_mask=mask)
            next_token = output[:, -1].argmax(-1)
            
            if next_token == word2id['<EOS>']:
                break
                
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            if print_:
                print(id2word[next_token.item()], end='', flush=True)
        
        if print_:
            print() # 换行
        return ''.join([id2word[c] for c in input_seq[0].tolist()])

# 示例使用
if __name__ == '__main__':
    word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")
    
    # 是否有模型就加载模型，没有就创建一个
    model =  load_model(Config.model_path) if os.path.exists(Config.model_path) else LeaperV1(100, len(word2id), 1, 1, 1, 1).to(device)
    
    while True:
        # 假设我们有一个开始token
        start_token = wordTotokens(input('prompt:')) + [word2id.get('<SEP>', word2id['<UNK>'])]

        # 生成文本
        generated_text = generate_text_wordByword(model, start_token, max_len=100, temperature=0.8, print_=True)