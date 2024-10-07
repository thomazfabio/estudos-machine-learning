Em muitos projetos de aprendizado de máquina, é comum dividir os dados em três conjuntos principais:

1. **Treinamento (Training set)**: Usado para ajustar os pesos da rede neural. Esse é o conjunto que mostramos para o modelo repetidamente durante o treinamento.
   
2. **Validação (Validation set)**: Usado para monitorar o desempenho do modelo durante o treinamento, ajustando hiperparâmetros e verificando se o modelo está aprendendo de forma consistente ou se está "decorando" os dados (overfitting). Esse conjunto não é usado diretamente no ajuste dos pesos.

3. **Teste (Test set)**: Usado **somente após o treinamento** para avaliar a performance final do modelo. Serve para verificar se o modelo generaliza bem para novos dados.

No exemplo anterior, usamos apenas o **conjunto de treino** e o **conjunto de teste**. Porém, em muitos casos, é uma boa prática adicionar um **conjunto de validação**. Como o MNIST não vem pré-dividido em três partes, você pode criar o conjunto de validação dividindo uma parte do conjunto de treino.

### Como Criar um Conjunto de Validação

Você pode separar parte dos dados de treinamento para validação. Vamos pegar uma pequena porcentagem (por exemplo, 20%) do conjunto de treino e usá-la como validação.

Aqui está como fazer isso:

```python
from torch.utils.data import random_split

# Tamanho do conjunto de validação (20% dos dados de treino)
val_size = int(0.2 * len(trainset))
train_size = len(trainset) - val_size

# Dividindo os dados de treino e validação
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# Criando DataLoaders para os conjuntos de treino e validação
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
```

### Como Usar o Conjunto de Validação Durante o Treinamento

Agora que temos um conjunto de validação, podemos medir o desempenho do modelo durante o treinamento. Após cada época, verificamos a perda e a precisão no conjunto de validação. Isso nos ajuda a monitorar se o modelo está "decorando" os dados de treino (overfitting) ou se está aprendendo de forma generalizada.

Aqui está como você pode atualizar o loop de treinamento para incluir a validação:

```python
for epoch in range(5):  # Número de épocas
    running_loss = 0.0
    
    # Fase de treinamento
    net.train()  # Colocar o modelo em modo de treino
    for images, labels in trainloader:
        optimizer.zero_grad()  # Zerar gradientes

        outputs = net(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calcula a perda
        loss.backward()  # Backward pass (calcula os gradientes)
        optimizer.step()  # Atualiza os pesos

        running_loss += loss.item()
    
    # Avaliando no conjunto de validação
    val_loss = 0.0
    correct = 0
    total = 0
    net.eval()  # Colocar o modelo em modo de avaliação (desliga o dropout, batchnorm, etc)
    with torch.no_grad():  # Não calcular gradientes durante a validação
        for images, labels in valloader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Época {epoch+1}, Perda de treino: {running_loss/len(trainloader)}, Perda de validação: {val_loss/len(valloader)}, Acurácia na validação: {accuracy}%")
```

### Resumo do Fluxo:
1. **Treinamento**: O modelo ajusta os pesos com base nos dados de treino.
2. **Validação**: Após cada época, o modelo é avaliado no conjunto de validação. Isso ajuda a monitorar se o modelo está overfitting.
3. **Teste**: O conjunto de teste é usado **somente no final**, para avaliar a performance do modelo após o término do treinamento.

### Por que é importante o conjunto de validação?

- **Monitoramento de overfitting**: Ele ajuda a detectar se o modelo está "decorando" os dados de treinamento, o que geralmente resulta em bom desempenho no conjunto de treino, mas baixo desempenho em dados novos (como o conjunto de teste).
- **Ajuste de hiperparâmetros**: Como a taxa de aprendizado, o número de épocas, o tamanho da rede, entre outros, podem ser ajustados com base no desempenho no conjunto de validação.