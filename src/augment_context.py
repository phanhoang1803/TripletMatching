import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_model(model_name, use_quantized=True, use_flash_attention=True):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, 
        quantization_config=BitsAndBytesConfig(load_in_4bit=use_quantized),
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        use_flash_attention_2=use_flash_attention
    )
    return model, processor

def generate_context(model, processor, image_path, captions, device):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare the conversation and prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate a brief context (4-5 sentences) for this image based on the following captions: " + " ".join(captions)},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Generate context
    output = model.generate(**inputs, max_new_tokens=200)
    
    # Decode and return the generated context
    return processor.decode(output[0], skip_special_tokens=True)

def augment_data(data, model, processor, base_image_path, device):
    for item in tqdm(data, desc="Generating context"):
        image_path = os.path.join(base_image_path, item['img_local_path'])
        if not os.path.exists(image_path):
            continue
        captions = [article['caption'] for article in item['articles']]
        item['context'] = generate_context(model, processor, image_path, captions, device)
        print("Context: ", item['context'])
    return data

def save_augmented_data(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)

def main(args):
    print("Loading data...")
    data = load_data(args.input_file)
    
    print("Setting up the model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = setup_model(args.model_name, args.use_quantized, args.use_flash_attention)
    model.to(device)
    
    print("Augmenting data with AI-generated context...")
    augmented_data = augment_data(data, model, processor, args.base_image_path, device)
    
    print("Saving augmented data...")
    save_augmented_data(augmented_data, args.output_file)
    
    print(f"Augmented dataset saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with AI-generated context")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, default='augmented_annotations.json', help="Path to save the augmented JSON file")
    parser.add_argument("--base_image_path", type=str, required=True, help="Base path to the image directory")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Name of the LLaVA model to use")
    parser.add_argument("--use_quantized", action='store_true', help="Use quantized model")
    parser.add_argument("--use_flash_attention", action='store_true', help="Use flash attention")
    
    args = parser.parse_args()
    main(args)
