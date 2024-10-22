import os
from PIL import Image

# Defina o caminho da pasta com as imagens a serem convertidas
input_folder = "/home/thomaz/projetos/estudos-machine-learning/datasets/coco/snake-v1/test/data/"
output_folder = "/home/thomaz/projetos/estudos-machine-learning/datasets/coco/snake-v1/test/data/convertido/"

# Certifique-se de que a pasta de saída existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicialize um contador para nomear os arquivos
counter = 0

# Percorra todos os arquivos na pasta de entrada
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Verifique se é um arquivo
    if os.path.isfile(file_path):
        # Defina o novo nome para o arquivo
        new_filename = f'teste-{counter:03d}.jpg'
        output_path = os.path.join(output_folder, new_filename)

        # Verifique se o arquivo é uma imagem que precisa de conversão
        try:
            with Image.open(file_path) as img:
                # Converta e salve a imagem como JPG
                img.convert('RGB').save(output_path, 'JPEG')
                print(f'{filename} convertido e salvo como {new_filename}')
        except Exception as e:
            # Se não for uma imagem, apenas copie e renomeie
            os.rename(file_path, output_path)
            print(f'{filename} renomeado para {new_filename}')
        
        counter += 1
