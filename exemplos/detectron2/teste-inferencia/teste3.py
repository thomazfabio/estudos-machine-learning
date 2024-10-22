import os
import random
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")

# Configuração do modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/home/thomaz/projetos/estudos-machine-learning/modelos-treinados/snake-v1/model_final.pth"  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Número de classes no dataset (Snake)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confiança mínima
cfg.MODEL.DEVICE = "cuda"  # Usando GPU

# Pasta com imagens de teste
pasta_teste = "/home/thomaz/projetos/estudos-machine-learning/datasets/coco/snake-v1/test/data/"

# Lista de imagens de teste
imagens_teste = [f for f in os.listdir(pasta_teste) if f.endswith('.jpg') or f.endswith('.png')]

# Função para escolher imagem aleatória
def get_imagem_aleatoria():
    return os.path.join(pasta_teste, random.choice(imagens_teste))

# Crie um objeto Predictor
predictor = DefaultPredictor(cfg)

# Configuração da opacidade (entre 0 e 1)
opacidade = 0.5  # 50% de transparência

# Espessura da borda em torno das máscaras
espessura_borda = 4  # Aumentando a espessura para 4

mostrar_imagens = True
while mostrar_imagens:
    try:
        # Escolha uma imagem aleatória
        imagem_path = get_imagem_aleatoria()
        print(f"Carregando imagem: {imagem_path}")
        
        # Carregue a imagem
        imagem = cv2.imread(imagem_path)
        if imagem is None:
            print(f"Erro: não foi possível carregar a imagem {imagem_path}")
            continue
        
        print(f"Dimensões da imagem: {imagem.shape}")
        
        # Faça a inferência
        print("Fazendo inferência na imagem...")
        outputs = predictor(imagem)
        print("Inferência concluída.")

        # Verifique se há instâncias detectadas
        if len(outputs["instances"]) == 0:
            print("Nenhuma instância detectada.")
            continue
        
        # Obtenha as máscaras das instâncias
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()
        print(f"Máscaras detectadas: {len(pred_masks)}")
        
        # Crie uma cópia da imagem para visualização
        visualizado = imagem.copy()

        # Processar e desenhar cada máscara
        for idx, mask in enumerate(pred_masks):
            print(f"Processando máscara {idx+1}...")
            
            # Converte a máscara para uma imagem binária e cria uma cor aleatória
            mask_img = mask.astype(np.uint8) * 255
            color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

            # Crie uma máscara colorida da mesma dimensão da imagem original
            mask_color = np.zeros_like(imagem, dtype=np.uint8)
            mask_color[mask == 1] = color

            # Desenha a borda azul ao redor da máscara
            contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                print(f"Desenhando borda azul da máscara {idx+1}...")
                cv2.drawContours(visualizado, contours, -1, (255, 0, 0), thickness=espessura_borda)  # Borda azul mais grossa
            
            # Mescle a imagem original com a máscara colorida (aplica a opacidade)
            visualizado = cv2.addWeighted(visualizado, 1.0, mask_color, opacidade, 0)

        # Redimensionar a janela para visualizar melhor
        window_name = "Resultado com Máscaras e Borda Azul"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Permite redimensionar a janela
        cv2.imshow(window_name, visualizado)
        cv2.resizeWindow(window_name, 800, 600)  # Ajusta o tamanho da janela para 800x600

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):  
            mostrar_imagens = False

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
