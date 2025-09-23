import os, time, argparse
from PIL import Image
import numpy as np
from openai import OpenAI
import joblib
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from utils.utils import print_args, load_restore_ckpt, load_embedder_ckpt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import base64

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

device_clip = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device_clip)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_feature(img_path):
    """Trích vector đặc trưng từ ảnh bằng CLIP"""
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device_clip)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.cpu().numpy().flatten()

# ========================
# Dataset + Classifier
# ========================
def build_dataset(root_dir):
    """Trích toàn bộ features cho dataset"""
    X, y, classes = [], [], sorted(os.listdir(root_dir))
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            feat = extract_feature(img_path)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y), classes


def load_or_train_classifier(train_dir, test_dir, model_path="clip_degradation_classifier.pkl"):
    """Nếu có .pkl thì load, ngược lại train mới, nhưng luôn in classification_report trên test set"""
    if os.path.exists(model_path):
        print(f"✅ Found existing model at {model_path}, loading...")
        clf, classes = joblib.load(model_path)

        # Dù load sẵn thì vẫn evaluate lại
        _, _, classes_test = build_dataset(test_dir)
        X_test, y_test, _ = build_dataset(test_dir)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=classes))
    else:
        print("⚡ Training new model...")
        X_train, y_train, classes = build_dataset(train_dir)
        X_test, y_test, _ = build_dataset(test_dir)

        print("Train size:", X_train.shape, " Test size:", X_test.shape)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)

        # Save model
        joblib.dump((clf, classes), model_path)

        # Evaluate
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=classes))

    return clf, classes


def predict_image(img_path, clf, classes):
    """Dự đoán class degradation cho ảnh bất kỳ"""
    feat = extract_feature(img_path)
    pred = clf.predict([feat])[0]
    return classes[pred]



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


# ========================
# Main pipeline
# ========================
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 1) Initialize models
    print('> Model Initialization...')
    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=args.restore_model_path)

    # --- 2) Load or train degradation classifier
    clf, classes = load_or_train_classifier(
        args.train_dir, args.test_dir, args.clip_classifier_path
    )

    os.makedirs(args.output, exist_ok=True)
    files = os.listdir(args.input)
    time_record = []

    for i in files:
        lq_path = os.path.join(args.input, i)
        lq = Image.open(lq_path)

        with torch.no_grad():
            # --- 3) Preprocess
            lq_re = torch.from_numpy((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            lq_em = transform_resize(lq).unsqueeze(0).to(device)

            start_time = time.time()

            # --- 4) Decide embedding source
            if args.prompt is None:
                # Step 1: predict degradation category bằng CLIP+LogReg
                pred_category_from_image = predict_image(lq_path, clf, classes)
                print(f'Estimated degradation (from image): {pred_category_from_image}')

                # Step 2: generate caption động
                caption = generate_caption(lq_path, pred_category_from_image, i)
                print(f'Generated caption: {caption}')

                # Step 3: encode caption thành text embedding
                text_embedding_caption = embedder([caption], 'text_encoder')
                used_text_embedding = text_embedding_caption
            else:
                # User provided a manual prompt
                text_embedding_prompt, _, [pred_category_from_prompt] = embedder([args.prompt], 'text_encoder')
                used_text_embedding = text_embedding_prompt
                print(f'Using user-provided prompt: "{args.prompt}" (category: {pred_category_from_prompt})')

            # --- 5) Run restoration
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
    
    parser.add_argument("--train_dir", type=str, required=True, help="Training dataset folder")
    parser.add_argument("--test_dir", type=str, required=True, help="Testing dataset folder")
    parser.add_argument("--clip_classifier_path", type=str, default="clip_degradation_classifier.pkl", help="Path to save/load classifier")
    
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