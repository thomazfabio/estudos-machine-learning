import torch

# Criando um tensor escalar (0D)
tensor_esc = torch.tensor(5)
print("Tensor Escalar:", tensor_esc)

# Criando um tensor vetor (1D)
tensor_vec = torch.tensor([1, 2, 3])
print("Tensor Vetor (1D):", tensor_vec)

# Criando um tensor matriz (2D)
tensor_mat = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor Matriz (2D):", tensor_mat)

# Criando um tensor tridimensional (3D)
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor 3D:", tensor_3d)

# Somando tensores
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([4, 5, 6])
soma = tensor_a + tensor_b
print("Soma de Tensores:", soma)

# Multiplicando tensores
produto = tensor_a * tensor_b
print("Multiplicação de Tensores:", produto)

# Mudando a forma do tensor (reshape)
tensor_original = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_reshaped = tensor_original.view(3, 2)  # Mudando a forma para 3 linhas, 2 colunas
print("Tensor Reshape (mudando a forma):", tensor_reshaped)

# Verificando se a GPU está disponível
if torch.cuda.is_available():
    device = torch.device("cuda")  # Usar GPU
    print("GPU disponível, usando CUDA")

    # Criando um tensor e movendo para GPU
    tensor_gpu = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print("Tensor na GPU:", tensor_gpu)

    # Operação na GPU
    tensor_gpu_squared = tensor_gpu ** 2
    print("Tensor elevado ao quadrado na GPU:", tensor_gpu_squared)

else:
    print("CUDA não disponível, usando CPU")