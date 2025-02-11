import torch
from torch import nn, optim
from torchvision import models

def build_model(arch, hidden_units):
    # Load a pre-trained model
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    if arch == "vgg13":
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[1], 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == "resnet34":
        model.fc = nn.Sequential(
            nn.Linear(512, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units[1], 102),
            nn.LogSoftmax(dim=1)
        )

    return model

def train_model(model, trainloader, validloader, epochs, learning_rate, use_gpu):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if hasattr(model, 'classifier') else optim.Adam(model.fc.parameters(), lr=learning_rate))

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

def save_checkpoint(model, save_dir, arch, hidden_units):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(model, image, top_k, use_gpu):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)

    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]