import torch
from tqdm import tqdm
from net import net
from loss import Loss


def train(
    train_loader,
    eval_loader,
    learning_rate,
    nepoch,
    patience,
    factor,
    earlystop,
    device,
):
    torch.manual_seed(0)
    model = net().to(device)
    criterion = Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )
    stop = 0
    best_dice = 0.0
    best_loss = float("inf")
    model.train()
    for epoch in range(nepoch):
        print(f"Epoch {epoch+1}")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            for imgs, masks, labels in batch:
                imgs, masks = imgs.to(device), masks.to(device)
                labels = labels.to(device)
                preds, pred_label = model(imgs)
                loss = criterion(preds, pred_label, masks, labels)
                loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                for imgs, masks, labels in batch:
                    imgs, masks = imgs.to(device), masks.to(device)
                    labels = labels.to(device)
                    preds, pred_label = model(imgs)
                    loss = criterion(preds, pred_label, masks, labels)
                    total_loss += loss.item() * imgs.size(0)
                    total_dice += criterion.dice * imgs.size(0)
                    total_acc += criterion.acc * imgs.size(0)
        test_loss = total_loss / len(eval_loader.dataset) / 2
        test_dice = total_dice / len(eval_loader.dataset)
        test_acc = total_acc / len(eval_loader.dataset) / 2
        scheduler.step(test_loss)
        print(f"loss: {test_loss}\tdice: {test_dice}\tacc: {test_acc}")
        # print(model.thresh)
        if test_dice > best_dice:
            best_dice = test_dice
            torch.save(model.state_dict(), "model.pth")
        if test_loss < best_loss:
            best_loss = test_loss
            stop = 0
        else:
            stop += 1
        if stop >= earlystop:
            print("Early Stopping!")
            break
    model.load_state_dict(torch.load("model.pth", map_location=device))
    torch.save(model, "model.pth")


def test(test_loader, device):
    print("Testing...")
    model = torch.load("model.pth", weights_only=False, map_location=device)
    criterion = Loss().to(device)
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for imgs, masks, labels in batch:
                imgs, masks = imgs.to(device), masks.to(device)
                labels = labels.to(device)
                preds, pred_label = model(imgs)
                loss = criterion(preds, pred_label, masks, labels)
                total_loss += loss.item() * imgs.size(0)
                total_dice += criterion.dice * imgs.size(0)
                total_acc += criterion.acc * imgs.size(0)
    test_loss = total_loss / len(test_loader.dataset) / 2
    test_dice = total_dice / len(test_loader.dataset)
    test_acc = total_acc / len(test_loader.dataset) / 2
    print(f"loss: {test_loss}\tdice: {test_dice}\tacc: {test_acc}")
