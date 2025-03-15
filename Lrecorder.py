# 这个文件录入对话数据
import json, os, rich, copy
import requests
from tqdm import tqdm

url = "https://api.siliconflow.cn/v1/chat/completions"

# TODO:reponse用api生成
def getResponse(prompt):
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": "Bearer sk-xyoshyicrwpnanyvdnghnnjbsyikpajzybiypfqkybhwenoq",
        "Content-Type": "application/json"
    }

    result = requests.request("POST", url, json=payload, headers=headers).json()['choices'][0]['message']['content']
    rich.print('[green]' + result)
    return result

def getTextBlock(prompt, end_prompt):
    text = str()
    rich.print(f'[blue]TextBlock-[red]{prompt} [green]输入[red]{end_prompt}[green]结束')
    while True:
        tmp = input(f'{prompt}>')
        if tmp == end_prompt:
            break
        text += tmp + '\n'
    return text[:-1] # 去掉最后一个换行符

def EditChatData(datafile, tokenfile, end_prompt):
    if os.path.exists(tokenfile):
        with open(datafile, 'r', encoding='utf-8') as f:
            chats = json.loads(f.read())
            print('Load chats from datas.json, length:', len(chats))
    else:
        chats = []
    while True:
        # 多轮对话
        history = []
        while True:
            query, response = getTextBlock('query', end_prompt), getTextBlock('response', end_prompt)
            history.append(query)
            chats.append({
                'history': copy.deepcopy(history), # 只有一轮对话
                'response': response
            })
            history.append(response)
            if input('继续这轮对话? (y/n)') == 'n':
                break
        if input('再来一轮对话? (y/n)') == 'n':
            break
    # 保存到文件
    with open(datafile, 'w+', encoding='utf-8') as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)
    with open(tokenfile, 'w+', encoding='utf-8') as f:
        for chat in chats:
            f.write('\n'.join(chat['history'])+'\n'+chat['response']+'\n')
    print('Chats already saved to datas.json, length:', len(chats))

def EditChatDataWithAI(datafile, tokenfile, end_prompt):
    if os.path.exists(tokenfile):
        with open(datafile, 'r', encoding='utf-8') as f:
            chats = json.loads(f.read())
            print('Load chats from datas.json, length:', len(chats))
    else:
        chats = []
    while True:
        # 多轮对话
        history = []
        while True:
            query = getTextBlock('query', end_prompt)
            response = getResponse(query)
            history.append(query)
            chats.append({
                'history': copy.deepcopy(history), # 只有一轮对话
                'response': response
            })
            history.append(response)
            if input('继续这轮对话? (y/n)') == 'n':
                break
        if input('再来一轮对话? (y/n)') == 'n':
            break
    # 保存到文件
    with open(datafile, 'w+', encoding='utf-8') as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)
    with open(tokenfile, 'w+', encoding='utf-8') as f:
        for chat in chats:
            f.write('\n'.join(chat['history'])+'\n'+chat['response']+'\n')
    print('Chats already saved to datas.json, length:', len(chats))

def EditChatDataFromJsonl(jsonl, datafile, tokenfile, end_prompt):
    if os.path.exists(tokenfile):
        with open(datafile, 'r', encoding='utf-8') as f:
            chats = json.loads(f.read())
            print('Load chats from datas.json, length:', len(chats))
    else:
        chats = []
    with open(jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            # 多轮对话
            line = json.loads(line.strip())['dialogue']
            i = 0
            
            # 获取一个对话
            def get_one(i):
                tmp = line[i]
                return tmp, i + 1

            history = []
            while len(line) > i:
                query, i = get_one(i)
                response, i = get_one(i)
                history.append(query)
                chats.append({
                    'history': copy.deepcopy(history), # 只有一轮对话
                    'response': response
                })
                history.append(response)
    # 保存到文件
    with open(datafile, 'w+', encoding='utf-8') as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)
    with open(tokenfile, 'w+', encoding='utf-8') as f:
        for chat in chats:
            f.write('\n'.join(chat['history'])+'\n'+chat['response']+'\n')
    print('Chats already saved to datas.json, length:', len(chats))

if __name__ == '__main__':
    EditChatDataFromJsonl('multiturn_cn_release_051623.jsonl', 'datas.json', 'traindata.txt', '<')