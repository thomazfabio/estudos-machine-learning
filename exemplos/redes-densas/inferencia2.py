import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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

# Função para carregar e processar a imagem
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Converte para escala de cinza
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Redimensiona para 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona uma dimensão de batch
    return image_tensor

# Caminho para a imagem JPEG
image_path = '/home/thomaz/projetos/estudos-machine-learning/imagens-e-videos/imagens/digitos-manuscritos/9.png'  # Substitua pelo caminho da sua imagem
image_tensor = load_image(image_path)  # Carregar e processar a imagem

# Fazer a inferência
with torch.no_grad():  # Desativa a gravação do gradiente
    output = net(image_tensor)
    _, predicted = torch.max(output, 1)  # Obter a classe prevista

# Visualizar a imagem original
original_image = Image.open(image_path).convert('L')
plt.imshow(original_image, cmap='gray')  # Exibe a imagem original
plt.title(f'Classe prevista: {predicted.item()}')
plt.axis('off')  # Remove os eixos
plt.show()