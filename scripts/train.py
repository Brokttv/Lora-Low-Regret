import torch


def train(model, optimizer, train_dataloader, device):
    model.train()
    dataset = train_dataloader
    num_batches = len(train_dataloader)
    total_loss = 0

    for batch in dataset:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / num_batches
    return train_loss


def val(model, val_dataloader, device):
    with torch.inference_mode():
        model.eval()
        dataset = val_dataloader
        num_batches = len(dataset)
        loss = 0

        for batch in dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss.item()

    val_loss = loss / num_batches
    return val_loss


def test(model, test_dataloader, device):
    with torch.inference_mode():
        model.eval()
        correct = 0
        total = 0

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    accuracy = correct / total
    return accuracy
