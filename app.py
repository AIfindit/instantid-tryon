import os
from flask import Flask, request, send_file, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from diffusers import DDIMScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from models.instantid.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from utils import encode_face

app = Flask(__name__)

# Inicializa o analisador facial com GPU
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0)

# Caminhos para os modelos locais
FACE_MODEL_PATH = "checkpoints/ip-adapter/ip-adapter-plus-face_sd15.bin"
CONTROLNET_PATH = "checkpoints/ControlNetModel"

# Carregamento do modelo ControlNet
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_PATH,
    torch_dtype=torch.float16
).to("cuda")

# Carregamento do pipeline com InstantID
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Configurações do IP-Adapter
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_ip_adapter_scale(0.6)
pipe.image_proj_model = CLIPVisionModelWithProjection.from_pretrained(
    "checkpoints/ControlNetModel/IP-Adapter", torch_dtype=torch.float16
).to("cuda")
pipe.image_processor = CLIPImageProcessor.from_pretrained("checkpoints/ControlNetModel/IP-Adapter")
@app.route("/tryon", methods=["POST"])
def tryon():
    try:
        image_file = request.files["image"]
        reference_file = request.files["reference"]
        prompt = request.form.get("prompt", "A realistic outfit")

        image_path = "input.jpg"
        reference_path = "reference.jpg"
        image_file.save(image_path)
        reference_file.save(reference_path)

        # Redimensiona mantendo a altura de 512px, largura proporcional
        original_image = Image.open(image_path).convert("RGB")
        w, h = original_image.size
        new_h = 512
        new_w = int(w * (512 / h))
        image = original_image.resize((new_w, new_h))

        reference = cv2.imread(reference_path)
        faces = face_analyzer.get(reference)
        if not faces:
            return jsonify({"error": "Nenhum rosto detetado"}), 400

        face_emb = encode_face(reference, faces[0])

        output = pipe(
            prompt=prompt,
            image=image,
            control_image=image,
            ip_adapter_image=face_emb,
            num_inference_steps=30
        ).images[0]

        result_path = "result.jpg"
        output.save(result_path)
        return send_file(result_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
