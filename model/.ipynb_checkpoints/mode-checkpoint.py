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
    data_all = []
    target_all = []
    output_all = []

    for epoch in range(epochs):
        
        print(f'Epoch {epoch+1}/{epochs}')
        
        
        for phase in ['train', 'val']:
            start = time.time()
            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0

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
                        
                        if (phase == 'val') and ((epoch+1)==epochs):
                            data_all.append(data.cpu().numpy())
                            target_all.append(target.cpu().numpy())
                            output_all.append(output.cpu().numpy())

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
    
    data_all = np.concatenate(data_all)
    target_all = np.concatenate(target_all)
    output_all = np.concatenate(output_all)
    result.plot_result(target_all.reshape(-1), output_all.reshape(-1))        
        # torch.save({'epoch': epoch,
        #             'model_state_dict': net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss,
        #            }, '256_middle.pth')

    train_time = sum(train_time_list) / len(train_time_list)
    val_time = sum(val_time_list) / len(val_time_list)
    
    return train_loss_list, val_loss_list, data_all, target_all, output_all, train_time, val_time



def test(device, model, dataloader, criterion):
    model.to(device)
    start = time.time()
    data_all = []
    target_all = []
    output_all = []
    loss_all = []
    test_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            low_seq, accessibility = batch
            data = low_seq.to(device, non_blocking=False)
            target = accessibility.to(device, non_blocking=False)

            output = model(data)

            data_all.append(data.cpu().detach().numpy())
            target_all.append(target.cpu().detach().numpy())
            output_all.append(output.cpu().detach().numpy())

            loss = criterion(output, target)
            loss_all.append(loss.item())


            test_loss += loss.item() * data.size(0)
    avg_loss = test_loss / len(dataloader.dataset)

    finish = time.time()
    test_time = finish - start
    print(f'Loss:{avg_loss:.4f} Timer:{test_time:.4f}')

    data_all = np.concatenate(data_all)
    target_all = np.concatenate(target_all)
    output_all = np.concatenate(output_all)
    
    return data_all, target_all, output_all, loss_all, test_time


def predict(device, net, input_seq):
    net.eval()
    data_all = []
    output_all = []
    with torch.no_grad():
        for data in input_seq:
            data = data.to(device)
            output = net(data)
    return data.cpu().numpy(), output.cpu().numpy()