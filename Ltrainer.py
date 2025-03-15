# 这个文件实现了训练模型的逻辑
from Lmodel import LeaperV1
import Ltokenizer
import argparse, torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
import Ldata
import Config, os
from tqdm import tqdm

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print('Model on', device_type)
device = torch.device(device_type)
word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")

def train(datafile, epochs, batch_size, lr, num_workers, size, shuffle=True):
    # 动态获取词典大小
    vocab_size = len(word2id)
    model = LeaperV1(
        max_len=Config.max_len,
        vocab_size=vocab_size,
        n_head=Config.n_head,
        layer_num=Config.layer_num,
        d_model=Config.d_model,
        dim_feedforward=Config.feedforward_dim
    ).to(device)
    print(model)
    # 检查是否已存在权重文件
    if os.path.exists(Config.model_path):
        print('Loaded model from', Config.model_path)
        model.load_state_dict(torch.load(Config.model_path))

    # 初始化模型、优化器和损失函数
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2id['<PAD>'])
    scaler = GradScaler()  # 混合精度训练支持

    # 创建数据加载器
    dataloader = Ldata.getDataLoader(
        datafile,
        batch_size,
        Config.max_len,
        num_workers,
        size,
        shuffle
    )

    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, targets, lengths in progress_bar:
            # 准备数据
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            padding_mask = (inputs == word2id['<PAD>'])
            
            # 生成因果掩码
            seq_len = inputs.size(1)
            causal_mask = model.generate_mask(seq_len).to(device)
            
            # 前向传播（混合精度）
            optimizer.zero_grad()
            with autocast(device_type):
                outputs = model(
                    inputs,
                    src_mask=causal_mask,
                    src_key_padding_mask=padding_mask
                )
                # 计算损失（展平为二维张量）
                loss = loss_fn(
                    outputs.reshape(-1, vocab_size),
                    targets.reshape(-1)
                )
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # 打印epoch统计信息
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # 定期保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), Config.model_path)
    
        # 保存最好模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.model_path)
            print(f'Model saved as {Config.model_path} at best loss:', best_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='datas.json', help='DataFile')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    # 训练对话时，batch_size可以改小一点，因为会占用大量内存(3左右)，模型的max_len也会影响pad后的数据大小
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--size', type=int, default=-1, help='Chat Data Size(-1 to use all)')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle')
    args = parser.parse_args()

    print('Data Length:', args.size)
    train(
        args.datafile,
        args.epochs,
        args.batch_size,
        args.lr,
        args.num_workers,
        args.size,
        args.shuffle
    )