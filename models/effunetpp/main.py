from loader import loader
from trainer import train, test

if __name__ == "__main__":
    data = loader(root="CocoGlide", batch_size=32, num_workers=4)
    train(
        train_loader=data.train_loader,
        eval_loader=data.eval_loader,
        learning_rate=1e-3,
        nepoch=30,
        patience=2,
        factor=0.5,
        earlystop=5,
        device="xpu",
    )
    test(test_loader=data.test_loader, device="xpu")
