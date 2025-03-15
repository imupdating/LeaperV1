# 此文件实现一个字符级的分词器，提供保存至json文件的功能


# 实现命令行参数读取功能
import argparse, json

word2id, id2word = {}, []

# 把文本转换成token
cutText = list

def get_dict_from_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        word2id = data['word2id']
        id2word = data['id2word']
        return word2id, id2word

def add_word(word):
    if word not in word2id:
        id = len(word2id)
        word2id[word] = id
        id2word.append(word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='traindata.txt', help='input file')
    parser.add_argument('--output', type=str, default='tokens.json', help='output file')
    args = parser.parse_args()

    # 添加模型训练的特殊token
    add_word('<PAD>')
    add_word('<UNK>') # 这个token在模型训练过程中不会被训练
    add_word('<EOS>') # 标记为一轮对话的结束(模型应主动停止输出)
    add_word('<SEP>') # 分隔符

    # 读取文件
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = cutText(text)
        for c in tokens:
            add_word(c)
    
    print('Total words:', len(word2id))

    with open(args.output, 'w+', encoding='utf-8') as f:
        json.dump({'word2id': word2id, 'id2word': id2word}, f, ensure_ascii=False, indent=4)

    print('Saved to', args.output)