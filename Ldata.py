# 这个文件处理数据的预加载，有些张量还得动态生成
import Ltokenizer
import argparse, json
from torch.utils.data import Dataset, DataLoader
import torch

# history转换成token，就是把每句话中间加一个<EOF>，然后拼接成一个字符串
# 所以<EOF>是一个特殊的token，既用来表示生成的结束，又用来标记一轮对话的结束

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")
informed = False

# 截断的时候通知一下用户，以便其更改模型参数
def inform_user(msg):
    global informed
    if not informed:
        print(msg)
        informed = True

class TextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = [word2id.get(c, word2id['<UNK>']) for c in sentence[:-1]]
        target_ids = [word2id.get(c, word2id['<UNK>']) for c in sentence[1:]]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# 数据整理函数
def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    sorted_indices = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)
    
    sorted_inputs = [inputs[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]
    sorted_lengths = [lengths[i] for i in sorted_indices]
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(sorted_inputs, batch_first=True, padding_value=word2id['<PAD>'])
    padded_targets = torch.nn.utils.rnn.pad_sequence(sorted_targets, batch_first=True, padding_value=word2id['<PAD>'])
    return padded_inputs, padded_targets, sorted_lengths

class DataGet():
    def __init__(self, data_file, max_length):
        """
        Args:
            data_file (str): 包含对话数据的JSON文件路径
            max_length (int): 模型输入的最大序列长度（含特殊标记）
        """
        self.data = self.load_data(data_file)
        self.max_length = max_length
        self.sep_token = '<SEP>'
        self.pad_token = '<PAD>'

    def load_data(self, data_file):
        """加载并预处理数据"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        history = sample['history']
        response = sample['response']
        
        # 处理历史对话（每个历史条目后添加<SEP>）
        history_tokens = []
        for h in history:
            history_tokens.extend(Ltokenizer.cutText(h))
            history_tokens.append(self.sep_token)  # 直接保存ID
        
        # 处理回复内容
        response_tokens = Ltokenizer.cutText(response)
        
        # 合并历史和回复
        full_tokens = history_tokens + response_tokens + ['<EOS>', ]

        return full_tokens

def getDataLoader(data_file, batch_size, max_length, num_workers, size, shuffle=True):
    datagetor = DataGet(data_file, max_length)
    items = [datagetor.__getitem__(i) for i in range(len(datagetor))]
    print('Total samples:', len(datagetor), 'size:', size, 'loaded')
    dataset = TextDataset(items[:size])
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

def processOneTurnChatData(input, output):
    chats = []
    with open(input, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # 去除\n并分割
            query, response = line.strip().split('\t')
            chats.append({
                'history': [query, ], # 只有一轮对话
                'response': response
            })
    # 保存到文件
    with open(output, 'w+', encoding='utf-8') as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)
    print('Chats already saved to datas.json, length:', len(chats))

if __name__ == '__main__':
    dataloader = getDataLoader('datas.json', 1, 50, 0, True)
    print('<EOS> id:', word2id.get('<EOS>'))
    for batch in dataloader:
        print('sample:', batch)
        break
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='traindata.txt', help='Input file')
    parser.add_argument('--type', type=str, default='chat')
    parser.add_argument('--output', type=str, default='datas.json', help='Output file')
    args = parser.parse_args()
    if args.type == 'chat':
        processOneTurnChatData(args.input, args.output)