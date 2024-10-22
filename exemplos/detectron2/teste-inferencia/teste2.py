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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confiança mínima
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.7  # NMS
cfg.MODEL.ROI_HEADS.CLASSES = ["Snake"]  
cfg.MODEL.DEVICE = "cuda"  

# Pasta com imagens de teste
pasta_teste = "/home/thomaz/projetos/estudos-machine-learning/datasets/coco/snake-v1/test/data/"

# Lista de imagens de teste
imagens_teste = [f for f in os.listdir(pasta_teste) if f.endswith('.jpg') or f.endswith('.png')]

# Função para escolher imagem aleatória
def get_imagem_aleatoria():
    return os.path.join(pasta_teste, random.choice(imagens_teste))

# Crie um objeto Predictor
predictor = DefaultPredictor(cfg)

mostrar_imagens = True
while mostrar_imagens:
    # Escolha uma imagem aleatória
    imagem_path = get_imagem_aleatoria()
    
    # Carregue a imagem
    imagem = cv2.imread(imagem_path)
    
    # Faça a inferência
    outputs = predictor(imagem)
    
    # Imprima as caixas delimitadoras, classes e probabilidades
    for classe, caixa, prob in zip(outputs["instances"].pred_classes, outputs["instances"].pred_boxes, outputs["instances"].scores):
        print(f"Classe: {classe}, Probabilidade: {prob:.2f}, Caixa: {caixa}")


        
    # Exiba a imagem com caixas delimitadoras
    visualizado = cv2.imread(imagem_path)
    for caixa in outputs["instances"].pred_boxes:
        x1, y1, x2, y2 = int(caixa[0]), int(caixa[1]), int(caixa[2]), int(caixa[3])
        cv2.rectangle(visualizado, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imshow("Resultado", visualizado)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord('q'):  
        mostrar_imagens = False
    
