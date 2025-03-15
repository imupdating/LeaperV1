# 流程链
import os, argparse, Lmodel, Ltokenizer, Config, torch

run_system_command = os.system

def tryLoadModel():
    if not os.path.exists(Config.model_path):
        print('model not found, ok!')
        return
    word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")
    # 动态获取词典大小
    vocab_size = len(word2id)
    try:
        model = Lmodel.LeaperV1(
            max_len=Config.max_len,
            vocab_size=vocab_size,
            n_head=Config.n_head,
            layer_num=Config.layer_num,
            d_model=Config.d_model,
            dim_feedforward=Config.feedforward_dim
        )
        model.load_state_dict(torch.load(Config.model_path))
    except Exception as e:
        print(e)
        print('model not match, deleting......')
        os.remove(Config.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='t', help='ct(Check model file and Train)/t(Test)/to(Tokenizer)')
    args = parser.parse_args()
    task = args.task
    if task == 'ct':
        print('Checking model file')
        tryLoadModel()
        run_system_command('python Ltrainer.py')
    elif task == 't':
        print('Testing')
        run_system_command('python Lapi.py')
    elif task == 'to':
        print('Tokenizing for one turn chat data(from Lrecorder.py)')
        run_system_command('python Ltokenizer.py')
    elif task == 'tod':
        print('Tokenizing for one turn chat data(direct from traindata.txt)')
        run_system_command('python Ltokenizer.py')
        run_system_command('python Ldata.py')
    else:
        print('Please input the correct task.')