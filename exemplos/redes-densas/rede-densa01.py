import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Usando: {device_type}")

# Transformação para converter imagens em tensores e normalizar os valores dos pixels
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregar os dados de treino e teste do MNIST
trainset = torchvision.datasets.MNIST(root='/home/thomaz/projetos/estudos-machine-learning/datasets/MNIST', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='/home/thomaz/projetos/estudos-machine-learning/datasets/MNIST', train=False, download=True, transform=transform)

# Carregadores de dados (batch size 64)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
print("Dados de treino:", len(trainset))
print("Dados de teste:", len(testset))

# Definindo a rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camadas
        self.fc1 = nn.Linear(28 * 28, 128)  # Camada totalmente conectada
        self.fc2 = nn.Linear(128, 64)        # Camada intermediária
        self.fc3 = nn.Linear(64, 10)         # Camada de saída (10 classes para dígitos de 0 a 9)

    def forward(self, x):
        # Flatten (achatar) a imagem 28x28 em um vetor de 784 valores
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))        # ReLU como função de ativação
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                    # Saída sem ativação (será tratada na loss function)
        return x

net = Net()
net.to(device)

# Função de perda (loss function) e otimizador
criterion = nn.CrossEntropyLoss()  # Função de perda para classificação
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Otimizador Adam

# Loop de treinamento com estimativa de tempo geral
num_epochs = 5
total_batches = len(trainloader) * num_epochs  # Número total de batches
global_progress_bar = tqdm(total=total_batches, desc="Progresso Geral", unit="batch")

start_time = time.time()  # Início do treinamento para medir o tempo total

for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start_time = time.time()  # Tempo de início da época atual

    # Barra de progresso para a época atual
    epoch_progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=False)

    # Treinamento
    for images, labels in epoch_progress_bar:
        # Mover os dados para a GPU
        images, labels = images.to(device), labels.to(device)

        # Zerar os gradientes dos parâmetros
        optimizer.zero_grad()

        # Passagem para frente (forward)
        outputs = net(images)

        # Calcular a perda
        loss = criterion(outputs, labels)

        # Passagem para trás (backward) e otimização
        loss.backward()
        optimizer.step()

        # Acumular a perda
        running_loss += loss.item()

        # Atualizar a barra de progresso da época com a perda média
        epoch_progress_bar.set_postfix({'Loss': running_loss / (epoch_progress_bar.n + 1)})

        # Atualizar a barra de progresso global
        global_progress_bar.update(1)

    # Fim da época, calcular o tempo total da época
    epoch_time = time.time() - epoch_start_time
    print(f"Época {epoch + 1} finalizada, perda média: {running_loss / len(trainloader):.4f}, Tempo da época: {epoch_time:.2f} segundos")

# Tempo total de treinamento
total_time = time.time() - start_time
print(f"Treinamento completo em: {total_time:.2f} segundos.")

# Salvar o modelo treinado
model_path = '/home/thomaz/projetos/estudos-machine-learning/modelos-treinados/rede-densa01.pth'
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_path)

print(f'Modelo salvo em: {model_path}')