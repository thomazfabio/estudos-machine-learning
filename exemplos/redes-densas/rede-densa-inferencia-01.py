import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import random

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definindo a rede neural (mesmo modelo usado durante o treinamento)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Camada totalmente conectada
        self.fc2 = nn.Linear(128, 64)        # Camada intermediária
        self.fc3 = nn.Linear(64, 10)         # Camada de saída

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Saída sem ativação
        return x

# Carregar o modelo treinado
model_path = '/home/thomaz/projetos/estudos-machine-learning/modelos-treinados/rede-densa01.pth' # Caminho do arquivo do modelo
net = Net()  # Crie uma nova instância do modelo
net.to(device)  # Envie o modelo para a GPU ou CPU conforme necessário

# Carregar o estado do modelo
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()  # Colocar o modelo em modo de avaliação


# Carregar o conjunto de teste
testset = datasets.MNIST(root='/home/thomaz/projetos/estudos-machine-learning/datasets/MNIST', train=False, download=True, transform=None)  # Sem transformações
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# Randomizar a amostra de teste
random_index = random.randint(0, len(testset) - 1)
image, label = testset[random_index]  # Obter a imagem original e o rótulo

# Transformar a imagem para a entrada do modelo
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
image_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona uma dimensão de batch e envia para a GPU

# Fazer a inferência
with torch.no_grad():  # Desativa a gravação do gradiente
    output = net(image_tensor)
    _, predicted = torch.max(output, 1)  # Obter a classe prevista

# Visualizar a imagem original e a previsão
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')  # Exibe a imagem original
plt.title(f'Classe verdadeira: {label}, Classe prevista: {predicted.item()}')
plt.axis('off')  # Remove os eixos

plt.subplot(1, 2, 2)
plt.imshow(image_tensor.cpu().numpy().squeeze(), cmap='gray')  # Exibe a imagem normalizada
plt.title(' Imagem Normalizada')
plt.axis('off')  # Remove os eixos

plt.show()
