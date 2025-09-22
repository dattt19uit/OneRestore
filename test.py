import os, time, argparse
from PIL import Image
import numpy as np
from openai import OpenAI


import torch
from torchvision import transforms

from torchvision.utils import save_image as imwrite
from utils.utils import print_args, load_restore_ckpt, load_embedder_ckpt

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_caption(img_path, category, img_id):
    """
    Generate an objective caption using the raw image as input.
    The LLM analyzes perceptual degradations directly from the image.
    """
    prompt = f"""
    You are an image degradation analysis assistant.

    Task:
    Analyze the image "{img_id}" and explain why it belongs to the category "{category}".
    Base your explanation only on what you observe in the image.

    Guidelines:
    1. Evaluate key perceptual properties: brightness, contrast, texture, sharpness, and clarity. For each, briefly explain how it appears in the image.
    2. Describe observable degradations in simple, objective terms (e.g., blurring, dimness, loss of detail, washed-out colors) and explain their impact on visibility.
    3. Determine the severity level of "{category}" in the image (e.g., low, moderate, strong) and justify your choice.
    4. Keep the explanation concise: 3–4 sentences, around 80-100 words.
    5. Conclude with a justification that clearly links the observed degradations to the "{category}" label.
    6. Do not mention technical details such as embeddings, features, or statistics.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    try:
        image_base64 = encode_image(img_path)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
        )
        caption = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM error on {img_id}: {e}")
        caption = ""

    return caption


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('> Model Initialization...')

    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=args.restore_model_path)

    os.makedirs(args.output, exist_ok=True)

    files = [f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    time_record = []
    results = {}

    # set this flag (from CLI or default). If True, the caption embedding will be used for restoration.
    use_caption_embedding = getattr(args, "use_caption_embedding", False)

    for fname in files:
        img_path = os.path.join(args.input, fname)
        lq = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            # prepare tensors
            lq_re = torch.Tensor((np.array(lq) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
            lq_em = transform_resize(lq).unsqueeze(0).to(device)

            start_time = time.time()

            # --- 1) Image encoder: get embedding and predicted (static) category from image
            text_embedding_img, _, [pred_category_from_image] = embedder(lq_em, 'image_encoder')
            print(f'Predicted static category (from image): {pred_category_from_image}')

            # --- 2) Generate a dynamic caption (LLM) explaining predicted category
            caption = generate_caption(img_path, pred_category_from_image, fname)
            caption = caption if isinstance(caption, str) else (caption.get("caption") if isinstance(caption, dict) else str(caption))
            print(f'Generated caption: {caption}')

            # --- 3) Decide which embedding to use for the restorer
            if use_caption_embedding and caption:
                # If you explicitly want to use caption text as prompt for restoration:
                text_embedding_caption, _, [pred_category_from_caption] = embedder([caption], 'text_encoder')
                used_text_embedding = text_embedding_caption
                used_category_label = pred_category_from_caption
                print(f'Using caption-derived embedding; caption predicted category: {pred_category_from_caption}')
            else:
                # Default: use image-derived embedding (do NOT overwrite pred_category)
                used_text_embedding = text_embedding_img
                used_category_label = pred_category_from_image
                if use_caption_embedding:
                    # caption was empty / failed — fallback
                    print('Caption was empty/failure; falling back to image-derived embedding.')

            # --- 4) Restore image using the chosen embedding
            out = restorer(lq_re, used_text_embedding)

            run_time = time.time() - start_time
            time_record.append(run_time)

            if args.concat:
                out = torch.cat((lq_re, out), dim=3)

            # save restored image
            out_path = os.path.join(args.output, fname)
            imwrite(out, out_path, value_range=(0, 1))
            print(f'✔ {fname} processed in {run_time:.4f}s -> {out_path}')

            # store results (preserve both predictions so you can audit)
            results[fname] = {
                "pred_category_from_image": pred_category_from_image,
                "used_category_label": used_category_label,
                "caption": caption,
                "runtime_sec": round(run_time, 4)
            }

    # # write captions/metadata to JSON
    # json_out = os.path.join(args.output, "captions_and_meta.json")
    # with open(json_out, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Captions and metadata saved to {json_out}")
    print(f"✅ Average processing time: {np.mean(time_record):.4f}s")
            

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Running")

    # load model
    parser.add_argument("--embedder-model-path", type=str, default = "./ckpts/embedder_model.tar", help = 'embedder model path')
    parser.add_argument("--restore-model-path", type=str, default = "./ckpts/onerestore_cdd-11.tar", help = 'restore model path')

    # select model automatic (prompt=False) or manual (prompt=True, text={'clear', 'low', 'haze', 'rain', 'snow',\
    #                'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow'})
    parser.add_argument("--prompt", type=str, default = None, help = 'prompt')

    parser.add_argument("--input", type=str, default = "./image/", help = 'image path')
    parser.add_argument("--output", type=str, default = "./output/", help = 'output path')
    parser.add_argument("--concat", action='store_true', help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)

    main(argspar)