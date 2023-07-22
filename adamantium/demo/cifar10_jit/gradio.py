from typing import List, Tuple, Dict
from PIL import Image

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms

from adamantium import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.demo.ckpt_path

    # read the labels from the txt file
    with open(cfg.demo.labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.demo.ckpt_path}>")
    model = torch.jit.load(cfg.demo.ckpt_path)

    log.info(f"Loaded Model: {model}")

    # transform the incoming image to VIT input format

    image_transform = transforms.Compose(
        [
            transforms.Resize((cfg.demo.image_size, cfg.demo.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def recognize(image: Image) -> Dict[str, float]:
        if image is None:
            return None

        image = image_transform(image)
        image = torch.unsqueeze(image, 0)

        preds = model(image)
        preds = torch.nn.functional.softmax(preds, dim=1)

        preds = preds[0].tolist()
        labeled_preds = {labels[i]: preds[i] for i in range(10)}

        print(labeled_preds)
        return labeled_preds


    demo = gr.Interface(
        fn=recognize,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name= "0.0.0.0", server_port=8080, share=True)

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()