import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

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
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)  # Apenas transforma e retorna o tensor

# Função para visualizar as detecções
def visualize_detections(image, outputs):
    v = Visualizer(image, MetadataCatalog.get("my_dataset_val"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]  # Retorna a imagem em formato BGR

# Função para processar a imagem e realizar a inferência
def process_image(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform_image(image).to(device)  # Mova o tensor para o dispositivo

        # Adiciona a dimensão do batch
        tensor = tensor.unsqueeze(0)  # Agora adiciona a dimensão do batch

        # Log da forma do tensor após a transformação
        print(f"[DEBUG] Forma do tensor após transformação: {tensor.shape}")

        with torch.no_grad():
            # Cria uma entrada no formato esperado pelo modelo
            batched_inputs = [{"image": tensor}]  # Coloca o tensor diretamente no dicionário

            # Log da forma do dicionário de entrada
            print(f"[DEBUG] Dicionário de entrada (batched_inputs): {batched_inputs}")

            # Forma do tensor dentro do dicionário
            input_tensor_shape = batched_inputs[0]["image"].shape
            print(f"[DEBUG] Forma do tensor dentro do dicionário: {input_tensor_shape}")

            # Verifica a forma do tensor antes de passar para o modelo
            print(f"[DEBUG] Forma do tensor extraído para inferência: {input_tensor_shape}")

            # Realiza a inferência - removendo a dimensão extra aqui
            output = model([{"image": tensor.squeeze(0)}])  # Removendo a dimensão do batch adicional

        # Log da saída do modelo
        print(f"[DEBUG] Saída do modelo: {output}")

        # O modelo retorna uma lista, então usamos [0] para pegar o primeiro elemento
        return visualize_detections(np.array(image), output[0])
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None  # Retorna None para indicar erro


# Função principal
def main(image_paths):
    model = load_detectron_model(model_path)
    for image_path in image_paths:
        result_image = process_image(image_path, model)
        if result_image is not None:
            Image.fromarray(result_image).show()  # Exibe a imagem resultante

if __name__ == "__main__":
    image_paths = ["/home/thomaz/projetos/estudos-machine-learning/imagens-e-videos/imagens/106.jpeg"]
    main(image_paths)
