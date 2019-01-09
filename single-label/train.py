import time
import copy
import torch
# from torch.autograd import Variable


def train_model(model_conv,
                criterion,
                optimizer_conv,
                exp_lr_scheduler,
                dataloaders,
                dataset_sizes,
                writer,
                num_epochs,
                saved_model,
                device):
    since = time.time()

    best_model_wts = copy.deepcopy(model_conv.state_dict())
    best_acc = 0.0
    train_iter = 0
    val_iter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model_conv.train()  # Set model to training mode
            else:
                model_conv.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_conv.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_conv(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_conv.step()

                        if batch_num % 100 == 0:
                            print('batch: #{}, loss = {}'.format(batch_num, loss.item()))
                        batch_num += 1
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                writer.add_scalar('data/train_loss', train_loss, train_iter)
                writer.add_scalar('data/train_acc', train_acc, train_iter)
                writer.add_scalars('data/scalar_group', {'train_loss': train_loss,
                                                         'train_acc': train_acc,}, train_iter)
                train_iter += 1
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                writer.add_scalar('data/val_loss', val_loss, val_iter)
                writer.add_scalar('data/val_acc', val_acc, val_iter)
                writer.add_scalars('data/scalar_group', {'val_loss': val_loss,
                                                         'val_acc': val_acc, }, val_iter)
                val_iter += 1

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_conv.state_dict())
                torch.save(best_model_wts, saved_model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_conv.load_state_dict(best_model_wts)

    return model_conv
