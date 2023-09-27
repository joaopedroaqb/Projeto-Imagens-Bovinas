import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
# import splitfolders

torch.manual_seed(123)

# splitfolders.ratio("xray_dataset", ratio=(0.8, 0.1, 0.1))

#Base de dados
data_dir_train = 'train'
data_dir_test = 'test'

transform_train = transforms.Compose(
    [
      transforms.Resize([64,64]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
     transforms.Resize([64,64]),
     transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(data_dir_train, transform = transform_train)
test_dataset = datasets.ImageFolder(data_dir_test, transform = transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle=True)
# Contrução do Modelo
class classificador(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
    self.conv2 = nn.Conv2d(64, 64, (3,3))
    self.activation = nn.ReLU()
    self.bnorm = nn.BatchNorm2d(num_features=64)
    self.pool = nn.MaxPool2d(kernel_size=(2,2))
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
    self.linear2 = nn.Linear(256, 128)
    self.output = nn.Linear(128, 15)

  def forward(self, X):
    X = self.pool(self.bnorm(self.activation(self.conv1(X))))
    X = self.pool(self.bnorm(self.activation(self.conv2(X))))
    X = self.flatten(X)
    X = self.activation(self.linear1(X))
    X = self.activation(self.linear2(X))
    
    # Saída
    X = self.output(X)

    return X

net = classificador()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.0)

device = torch.device('cpu')
device

net.to(device)

def training_loop(loader, epoch):
    running_loss = 0.
    running_accuracy = 0.

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()        

        outputs = net(inputs)
        ps = F.softmax(outputs, dim=1)
        top_p, top_class = ps.topk(k = 1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.float))
        running_accuracy += accuracy
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        print('\rÉpoca {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end = '\r')
    
    # Imprimindo os dados referentes a essa época
    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss/len(loader), running_accuracy/len(loader)))

epochs = 30
for epoch in range(epochs):
  # Treino
  print("Treinando")
  training_loop(train_loader, epoch)
  net.eval()
  # Teste
  print("Validando")
  training_loop(test_loader, epoch)
  net.train()

net.eval()
torch.save(net.state_dict(), "checkpoint.pth")

