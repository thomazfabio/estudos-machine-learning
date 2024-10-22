import torch
from PIL import Image
from torchvision import transforms
import os
import random

imgPath = '/home/thomaz/projetos/estudos-machine-learning/datasets/coco/snake-v1/test/data/'

# Carregue o modelo treinado
model = torch.load('/home/thomaz/projetos/estudos-machine-learning/modelos-treinados/snake-v1/model_final.pth')

# Defina a função de inferência
def inferencia(imagem_path):
    # Carregue a imagem
    imagem = Image.open(imagem_path)
    
    # Transforme a imagem para o formato necessário pelo modelo
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagem = transform(imagem)
    
    # Faça a inferência
    output = model(imagem.unsqueeze(0))
    
    # Retorne o resultado
    return output.detach().numpy()

imagens = [f for f in os.listdir(imgPath) if f.endswith('.jpg') or f.endswith('.png')]
def get_imagem_aleatoria():
    return os.path.join(imgPath, random.choice(imagens))

# Exemplo de uso
imagem_path = get_imagem_aleatoria()
resultado = inferencia(imagem_path)
print(resultado)