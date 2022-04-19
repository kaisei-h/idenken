import torch
import time
import numpy as np
# 自作
import result

def train(device, net, dataloaders_dict, criterion, optimizer, epochs):
    net.to(device)
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    train_loss_list = []
    val_loss_list = []
    train_time_list = []
    val_time_list = []

    for epoch in range(epochs):
        
        print(f'Epoch {epoch+1}/{epochs}')
        
        
        for phase in ['train', 'val']:
            start = time.time()
            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0

            data_all = []
            target_all = []
            output_all = []
            if (epoch==0) and (phase=='train'):
                continue

            # for data, target in dataloaders_dict[phase]: # オリジナル用
            #     data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)

            for batch in dataloaders_dict[phase]: # RNABERT用
                low_seq, _, accessibility = batch
                data = low_seq.to(device, non_blocking=False)
                target = accessibility.to(device, non_blocking=False)


                optimizer.zero_grad()
                if data.size()[0] == 1:
                    continue
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase=='train'):
                        output = net(data)
                                
                        if (phase == 'val') and ((epoch+1)%5==0):
                            for i in range(len(target)):
                                # if (0 in data[i]):
                                #     output[i][:256-data[i].tolist().index(0)] = 0
                                # data_all.append(data[i].cpu().numpy())
                                target_all.append(target[i].cpu().numpy())
                                output_all.append(output[i].cpu().numpy())

                        loss = criterion(output, target)
                        if phase == 'train':
                            # loss.backward()
                            # optimizer.step()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        epoch_loss += loss.item() * data.size(0)
                        

            avg_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            
            finish = time.time()
            print(f'{phase} Loss:{avg_loss:.4f} Timer:{finish - start:.4f}')

            
            if phase=='val':
                val_time_list.append(finish - start)
                val_loss_list.append(avg_loss)
                if avg_loss<0.1:
                    break
            elif phase=='train':
                train_time_list.append(finish - start)
                train_loss_list.append(avg_loss)
                # scheduler.step()
        
        if ((epoch+1)%5==0):
            result.plot_result(np.array(target_all).reshape(-1), np.array(output_all).reshape(-1))
        
        # torch.save({'epoch': epoch,
        #             'model_state_dict': net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss,
        #            }, '256_middle.pth')
    train_time = sum(train_time_list) / len(train_time_list)
    val_time = sum(val_time_list) / len(val_time_list)
    
    return train_loss_list, val_loss_list, target_all, output_all, train_time, val_time


def test(device, net, test_dataloader, criterion):
    net.eval()
    data_all = []
    target_all = []
    output_all = []
    test_loss = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            for i in range(len(target)):
#                 if (0 in data[i]):
#                      output[i][:256-data[i].tolist().index(0)] = 0
                data_all.append(data[i].cpu().numpy())
                target_all.append(target[i].cpu().numpy())
                output_all.append(output[i].cpu().numpy())
            loss = criterion(output, target)
            test_loss += loss.item()
        test_loss = test_loss / len(test_dataloader)
    return test_loss, data_all, target_all, output_all


def predict(device, net, input_seq):
    net.eval()
    data_all = []
    output_all = []
    with torch.no_grad():
        for data in input_seq:
            data = data.to(device)
            output = net(data)
    return data.cpu().numpy(), output.cpu().numpy()