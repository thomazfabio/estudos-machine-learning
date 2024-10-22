import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import cv2

# Caminho para o modelo treinado
model_path = "/home/thomaz/projetos/estudos-machine-learning/modelos-treinados/inf/model_final.pth"

# Defina o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para carregar o modelo treinado
def load_detectron_model(model_path):
    cfg = get_cfg()
    cfg.merge_from_file("/home/thomaz/.local/lib/python3.10/site-packages/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Ajuste para o número de classes no seu dataset

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    print(f"[INFO] Modelo carregado: {model_path}")
    print(f"[INFO] Número de classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"[INFO] Classes disponíveis: ['pool']")

    return model.to(device)  # Mova o modelo para o dispositivo

# Função para transformar a imagem para tensor
def transform_image(image):
    transform = T.Compose([
        T.Resize((640, 640)),  # Redimensione a imagem para 640x640
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)  # Apenas transforma e retorna o tensor


# Função para visualizar as detecções
def visualize_detections(image, outputs):
    v = Visualizer(image, MetadataCatalog.get("my_dataset_val"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]  # Retorna a imagem em formato BGR

# Função para processar cada frame do vídeo e realizar a inferência
def process_frame(frame, model):
    try:
        # Converte o frame de BGR para RGB e depois para PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        
        # Transforma a imagem em tensor
        tensor = transform_image(image).to(device)  # Mova o tensor para o dispositivo

        # Adiciona a dimensão do batch
        tensor = tensor.unsqueeze(0)  # Agora adiciona a dimensão do batch, resultando em (1, 3, 640, 640)

        with torch.no_grad():
            # Cria uma entrada no formato esperado pelo modelo
            batched_inputs = [{"image": tensor}]  # Coloca o tensor diretamente no dicionário

            # Realiza a inferência
            output = model(batched_inputs)  # Usa o batched_inputs

        # Log da saída do modelo
        print(f"[DEBUG] Saída do modelo: {output}")

        return visualize_detections(frame, output[0])  # Visualiza a detecção no frame original
    except Exception as e:
        print(f"Erro ao processar o frame: {e}")
        return frame  # Retorna o frame original em caso de erro


# Função principal para captura de vídeo
def main(video_path):
    model = load_detectron_model(model_path)
    cap = cv2.VideoCapture(video_path)  # Abre o vídeo

    while cap.isOpened():
        ret, frame = cap.read()  # Lê um frame do vídeo
        if not ret:
            break  # Sai do loop se não conseguir ler o frame

        result_frame = process_frame(frame, model)  # Processa o frame

        cv2.imshow("Detecções", result_frame)  # Mostra o frame resultante
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Sai se a tecla 'q' for pressionada
            break

    cap.release()  # Libera o objeto de captura de vídeo
    cv2.destroyAllWindows()  # Fecha todas as janelas

if __name__ == "__main__":
    video_path = "/home/thomaz/projetos/estudos-machine-learning/imagens-e-videos/videos/pool-01.mp4"  # Insira o caminho do seu vídeo
    main(video_path)
