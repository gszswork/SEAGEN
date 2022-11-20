from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 通过创建不同的Summary Writer来写不同的model
    writer = SummaryWriter('SEAGEN/model1')
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x/score1', i * 2, i)

    writer1 = SummaryWriter('SEAGEN/model2')
    for i in x:
        writer1.add_scalar('y=2x/score1', i * 3, i)
    writer.close()