import torch
import detectron2
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
import os
import json
import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm

print(torch.__version__)
print(detectron2.__version__)

# Função para registrar o dataset no formato COCO
def get_dataset_dicts(img_dir):
    json_file = os.path.join(img_dir, "_annotations.coco.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in tqdm(enumerate(imgs_anns['images']), desc="Processando imagens"):
        record = {}

        filename = os.path.join(img_dir, v["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in imgs_anns['annotations'] if anno['image_id'] == v['id']]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"] -1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Registrar datasets de treino e validação
for d in ["train", "val"]:
    DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_dataset_dicts(f"/home/thomaz/projetos/estudos-machine-learning/datasets/coco/pool-detection-and-segmentation-coco-segmentation/{d}"))
    MetadataCatalog.get("my_dataset_" + d).set(thing_classes=["pool"])  # Ajuste para suas classes

# Configuração do modelo
cfg = get_cfg()
cfg.merge_from_file("/home/thomaz/.local/lib/python3.10/site-packages/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000  # Ajuste o número de iterações conforme necessário
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Ajuste para o número de classes no seu dataset

# Criação do Trainer personalizado para incluir barra de progresso
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Adiciona barra de progresso para o loader de treinamento
        data_loader = build_detection_train_loader(cfg)
        return tqdm(data_loader, desc="Treinando modelo")  # Removido o len()
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # Adiciona um hook para salvar checkpoints ao final de cada etapa
        hooks_list.insert(-1, hooks.PeriodicCheckpointer(self.checkpointer, period=5000))
        return hooks_list

# Iniciar o treinamento
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()