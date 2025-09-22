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

    # --- 1) Initialize models
    print('> Model Initialization...')
    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=args.restore_model_path)

    os.makedirs(args.output, exist_ok=True)

    files = os.listdir(args.input)
    time_record = []

    for i in files:
        lq_path = os.path.join(args.input, i)
        lq = Image.open(lq_path)

        with torch.no_grad():
            # --- 2) Preprocess
            lq_re = torch.Tensor((np.array(lq) / 255).transpose(2, 0, 1)).unsqueeze(0).to(device)
            lq_em = transform_resize(lq).unsqueeze(0).to(device)

            start_time = time.time()

            # --- 3) Decide embedding source
            if args.prompt is None:
                # Step 1: image encoder estimates degradation
                text_embedding_img, _, [pred_category_from_image] = embedder(lq_em, 'image_encoder')
                print(f'Estimated degradation (from image): {pred_category_from_image}')

                # Step 2: generate caption dynamically
                caption = generate_caption(lq_path, pred_category_from_image, i)
                print(f'Generated caption: {caption}')

                # Step 3: encode caption as dynamic prompt
                text_embedding_caption, _, [pred_category_from_caption] = embedder([caption], 'text_encoder')
                used_text_embedding = text_embedding_caption
                used_category_label = pred_category_from_caption
                print(f'Using caption-derived embedding; caption predicted category: {pred_category_from_caption}')
            else:
                # User provided a manual prompt
                text_embedding_prompt, _, [pred_category_from_prompt] = embedder([args.prompt], 'text_encoder')
                used_text_embedding = text_embedding_prompt
                used_category_label = pred_category_from_prompt
                print(f'Using user-provided prompt: "{args.prompt}" (category: {pred_category_from_prompt})')

            # --- 4) Run restoration
            out = restorer(lq_re, used_text_embedding)

            run_time = time.time() - start_time
            time_record.append(run_time)

            if args.concat:
                out = torch.cat((lq_re, out), dim=3)

            imwrite(out, os.path.join(args.output, i), value_range=(0, 1))
            print(f'{i} → Done. Running Time: {run_time:.4f}s.')

    print(f'Average time is {np.mean(time_record):.4f}s')
            

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