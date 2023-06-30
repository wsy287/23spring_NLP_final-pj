import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from tqdm import tqdm
from processing import classification_data,ComVEDataset,encode_data
from torch.utils.data import DataLoader
import os
import numpy as np
save_path='model/ALbert/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.cuda.empty_cache()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

version='albert-xxlarge-v2'
tokenizer=AutoTokenizer.from_pretrained(version)
model=AutoModelForMultipleChoice.from_pretrained(version).to(device)

lr=1e-5
batch_size=8
max_epoch=4
optimizer=torch.optim.Adam(model.parameters(), lr=lr)

# question='If the following statement is in common sense ?'
question='If the following statement is against common sense ?'
train_data=classification_data('train')
val_data=classification_data('dev')
test_data=classification_data('test')
train_encodings=encode_data(train_data,tokenizer,question)
val_encodings=encode_data(val_data,tokenizer,question)
test_encodings=encode_data(test_data,tokenizer,question)
train_dataset=ComVEDataset(train_encodings,train_data.iloc[:,3].values)
val_dataset=ComVEDataset(val_encodings,val_data.iloc[:,3].values)
test_dataset=ComVEDataset(test_encodings,test_data.iloc[:,3].values)

def validate(model):
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0
        record = list()

        pbar=tqdm(val_loader)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = output['loss']

            # make predictions
            predictions = output["logits"].argmax(dim=1)

            # count accuracy
            correct += predictions.eq(labels).sum().item()
            count += len(labels)
            accuracy = correct / count

            # show progress along with metrics
            pbar.set_postfix({
                'Loss': '{:.3f}'.format(loss.item()),
                'Accuracy': '{:.3f}'.format(accuracy)
            })

            # record the results
            record.append((int(labels),int(predictions)))
        pbar.close()
    print("The accuracy on the val_dataset: %.3f"%accuracy)
    return accuracy

# Train
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model.train()
print('Start training!')

train_loss = list()
train_accuracy = list()
valid_accuracy = list()
highest_accuracy = 0
highest_epoch = 0

for epoch in range(max_epoch):
    print('Epoch %s/%s' % (epoch + 1, max_epoch))

    correct = 0
    count = 0

    pbar = tqdm(train_loader)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        # make predictions
        predictions = output["logits"].argmax(dim=1)

        # count accuracy
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        accuracy = correct / count

        # show progress along with metrics
        pbar.set_postfix({
            'Loss': '{:.3f}'.format(loss.item()),
            'Accuracy': '{:.3f}'.format(accuracy)
        })

        # record the loss for each batch
        train_loss.append(loss.item())
        train_accuracy.append(accuracy)


    pbar.close()

    # record the loss and accuracy for each epoch
    # train_loss += epoch_loss
    # train_accuracy.append(accuracy)
    val_accuracy = validate(model)
    valid_accuracy.append(val_accuracy)

    # save the best model
    if highest_accuracy <= val_accuracy:
        highest_accuracy = val_accuracy
        highest_epoch = epoch
        save_dir = save_path + str(epoch) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir )
        torch.save(model.state_dict(), save_dir + 'albert_mc.pth')

np.save(save_path + 'train_acc.npy',np.array(train_accuracy))
np.save(save_path + 'val_acc.npy',np.array(valid_accuracy))
np.save(save_path + 'loss.npy',np.array(train_loss))
torch.cuda.empty_cache()

def test(model_path):
    model.load_state_dict(torch.load(model_path))
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
    model.eval()
    with torch.no_grad():
        correct=0
        count=0
        record=list()

        pbar=tqdm(test_loader)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = output['loss']

            # make predictions
            predictions = output["logits"].argmax(dim=1)

            # count accuracy
            correct += predictions.eq(labels).sum().item()
            count += len(labels)
            accuracy = correct / count

            # show progress along with metrics
            pbar.set_postfix({
                'Loss': '{:.3f}'.format(loss.item()),
                'Accuracy': '{:.3f}'.format(accuracy)
            })

            # record the results
            record.append((int(labels), int(predictions)))
        pbar.close()
        print("The accuracy on the test_dataset: %.3f" % accuracy)

test(save_dir+'albert_mc.pth')
