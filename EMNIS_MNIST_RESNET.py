
optimizer = optim.Adam(model_1.parameters(), lr=0.01)

def get_num_correct(preds,labels):
    
    return preds.argmax(dim = 1).eq(labels).sum().item()

epochs = 20
for epoch in range(epochs):

    #-----------------------------------Training loop----------------------------------------
    model_1.train()
    train_loss = 0
    train_correct = 0
    for batch in train_loader:                      # Get Batch
        
        images = batch[0].cuda()
        labels = batch[1].cuda()

        preds = model_1(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
        train_loss += loss.item()
        
        train_correct += get_num_correct(preds,labels)
    train_accuracy = train_correct/len(train_set)

    #-------------------------------------validation loop-------------------------------------
    model_1.eval()
    valid_loss =  0
    valid_correct = 0
    with torch.no_grad():

        for v_batch in valid_loader:                      # Get Batch
            
            v_images = v_batch[0].cuda()
            v_labels = v_batch[1].cuda()

            v_preds = model_1(v_images) # Pass Batch
            v_loss = F.cross_entropy(v_preds, v_labels) # Calculate Loss
            
            valid_loss += v_loss.item()


            valid_correct += get_num_correct(v_preds,v_labels)
        valid_accuracy = valid_correct/len(valid_set)

    print("epoch: ", epoch+1 , "train loss:", train_loss, "train_accuracy:" ,train_accuracy, "valid_loss:", valid_loss, "valid_accuracy", valid_accuracy )
