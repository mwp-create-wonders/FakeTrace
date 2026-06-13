import torch
from tqdm import tqdm
from os import remove
from net import net
from loss import Loss_mask, Loss_label


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
    model_mask = model.mask
    criterion_mask = Loss_mask().to(device)
    optimizer_mask = torch.optim.Adam(model_mask.parameters(), lr=learning_rate)
    scheduler_mask = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mask, mode="min", patience=patience, factor=factor
    )
    stop_mask = 0
    best_dice = 0.0
    best_loss = float("inf")
    for epoch in range(nepoch):
        print(f"Epoch {epoch+1}")
        model_mask.train()
        for batch in tqdm(train_loader):
            optimizer_mask.zero_grad()
            for imgs, masks, _ in batch:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model_mask(imgs)
                loss = criterion_mask(preds, masks)
                loss.backward()
            optimizer_mask.step()
        model_mask.eval()
        total_loss = 0.0
        total_dice = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                for imgs, masks, _ in batch:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = model_mask(imgs)
                    loss = criterion_mask(preds, masks)
                    total_loss += loss.item() * imgs.size(0)
                    total_dice += criterion_mask.dice * imgs.size(0)
        test_loss = total_loss / len(eval_loader.dataset) / 2
        test_dice = total_dice / len(eval_loader.dataset)
        scheduler_mask.step(test_loss)
        print(f"loss: {test_loss}\tdice: {test_dice}")
        if test_dice > best_dice:
            best_dice = test_dice
            torch.save(model_mask.state_dict(), "model_mask.pth")
        if test_loss < best_loss:
            best_loss = test_loss
            stop_mask = 0
        else:
            stop_mask += 1
        if stop_mask >= earlystop:
            print("Early Stopping!")
            break
    model_mask.load_state_dict(torch.load("model_mask.pth", map_location=device))
    model_mask.eval()
    train_pred = torch.tensor([], device=device)
    train_label = torch.tensor([], device=device)
    eval_pred = torch.tensor([], device=device)
    eval_label = torch.tensor([], device=device)
    with torch.no_grad():
        for batch in tqdm(train_loader):
            for imgs, masks, labels in batch:
                imgs, masks = imgs.to(device), masks.to(device)
                labels = labels.to(device)
                pred_mask = model_mask(imgs)
                pred_label = model.mask_to_label(pred_mask)
                train_pred = torch.cat((train_pred, pred_label), dim=0)
                train_label = torch.cat((train_label, labels), dim=0)
        for batch in eval_loader:
            for imgs, masks, labels in batch:
                imgs, masks = imgs.to(device), masks.to(device)
                labels = labels.to(device)
                pred_mask = model_mask(imgs)
                pred_label = model.mask_to_label(pred_mask)
                eval_pred = torch.cat((eval_pred, pred_label), dim=0)
                eval_label = torch.cat((eval_label, labels), dim=0)
    # for i in range(len(eval_label)):
    #     print(f"{eval_pred[i][0]}\t{eval_label[i]}")
    # exit()
    model_label = model.label
    criterion_label = Loss_label().to(device)
    optimizer_label = torch.optim.Adam(model_label.parameters(), lr=0.01)
    scheduler_label = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_label, mode="min", patience=patience, factor=factor
    )
    stop_label = 0
    best_acc = 0.0
    best_bce = float("inf")
    while True:
        model_label.train()
        pred_label = model_label(train_pred)
        loss = criterion_label(pred_label, train_label)
        optimizer_label.zero_grad()
        loss.backward()
        optimizer_label.step()
        model_label.eval()
        with torch.no_grad():
            pred_label = model_label(eval_pred)
            loss = criterion_label(pred_label, eval_label)
        scheduler_label.step(loss.item())
        print(f"loss: {loss.item()}\tacc: {criterion_label.acc}")
        if criterion_label.acc > best_acc:
            best_acc = criterion_label.acc
            torch.save(model_label.state_dict(), "model_label.pth")
        if loss.item() < best_bce:
            best_bce = loss.item()
            stop_label = 0
            # torch.save(model_label.state_dict(), "model_label.pth")
        else:
            stop_label += 1
        if stop_label >= earlystop:
            print("Early Stopping!")
            break
    model.mask.load_state_dict(torch.load("model_mask.pth", map_location=device))
    model.label.load_state_dict(torch.load("model_label.pth", map_location=device))
    torch.save(model, "model.pth")
    # remove("model_mask.pth")
    # remove("model_label.pth")


def test(test_loader, device):
    print("Testing...")
    model = torch.load("model.pth", weights_only=False, map_location=device)
    # print(model.label.weight,model.label.bias);exit()
    criterion_mask = Loss_mask().to(device)
    criterion_label = Loss_label().to(device)
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
                loss_mask = criterion_mask(preds, masks)
                loss_label = criterion_label(pred_label, labels)
                total_dice += criterion_mask.dice * imgs.size(0)
                total_acc += criterion_label.acc * imgs.size(0)
    test_dice = total_dice / len(test_loader.dataset)
    test_acc = total_acc / len(test_loader.dataset) / 2
    print(f"dice: {test_dice}\tacc: {test_acc}")
