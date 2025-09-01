import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm

class transformImg:
    def __init__(self, istrain):
        self.istrain = istrain

    def __call__(self, img):
        img = self.fixOrientation(img)
        if self.istrain and random.random() > 0.7:
            img = self.augmentImg(img)
        return img

    def fixOrientation(self, image):
        image = image.transpose(Image.ROTATE_90)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def augmentImg(self, image):
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.9, 1.1)
        translate_x = random.uniform(-0.1, 0.1) * 28
        translate_y = random.uniform(-0.1, 0.1) * 28

        return transforms.functional.affine(
            image,
            angle=angle,
            translate=(translate_x, translate_y),
            scale=scale,
            shear=0,
            fill=0
        )

def loadData():
    trainTransforms = transforms.Compose([
        transformImg(istrain=True),
        transforms.ToTensor()
    ])

    testTransforms = transforms.Compose([
        transformImg(istrain=False),
        transforms.ToTensor()
    ])

    forTrainDataLoading = datasets.EMNIST(
        root="./data",
        split='byclass',
        train=True,
        download=True,
        transform=trainTransforms
    )

    forTestDataLoading = datasets.EMNIST(
        root="./data",
        split='byclass',
        train=False,
        download=True,
        transform=testTransforms
    )

    trainData = DataLoader(forTrainDataLoading, shuffle=True, batch_size=64)
    testData = DataLoader(forTestDataLoading, shuffle=False, batch_size=64)

    return trainData, testData


class Network(nn.Module):
    def __init__(self, input, architecture):
        super(Network, self).__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        prev = input
        for layerSize in architecture:
            self.layers.append(nn.Linear(prev, layerSize))
            prev = layerSize

    def forward(self, x):
        x = self.flatten(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def saveModel(self, fileName):
        data = {}
        for i, layer in enumerate(self.layers):
            key = f"layer{i}" if i < len(self.layers) - 1 else f"outputlayer{i}"
            data[f"{key}weights"] = layer.weight.detach().cpu().numpy().tolist()
            data[f"{key}biases"] = layer.bias.detach().cpu().numpy().tolist()

        with open(fileName, "w") as file:
            json.dump(data, file)
        print(f" Model saved to {fileName}")


def evaluate(model, testingdata, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in testingdata:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    return accuracy


def train(model, costCalculator, optimizer, trainingdata, testingdata, epochs, device="cpu"):
    for epoch in range(epochs):
        model.train()
        total_cost_in_epoch = 0
        correct_in_epoch = 0
        total_in_epoch = 0

        progress_bar = tqdm(enumerate(trainingdata), total=len(trainingdata), desc=f"Epoch {epoch+1}/{epochs}")

        for batchNum, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            prediction = model(data)
            cost = costCalculator(prediction, target)
            cost.backward()
            optimizer.step()

            total_cost_in_epoch += cost.item()

            pred = prediction.argmax(dim=1)
            correct_in_epoch += (pred == target).sum().item()
            total_in_epoch += target.size(0)

            avg_loss = total_cost_in_epoch / (batchNum + 1)
            avg_acc = 100 * correct_in_epoch / total_in_epoch
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

        test_acc = evaluate(model, testingdata, device)
        print(f"\n Epoch {epoch+1} finished — Train Acc: {avg_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Network(input=784, architecture=[256, 128, 64, 62]).to(device)
    costCalculator = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainingData, testingData = loadData()
    train(model, costCalculator, optimizer, trainingData, testingData, epochs=10, device=device)

    model.saveModel("emnist_model.json")
