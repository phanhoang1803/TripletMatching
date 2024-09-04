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
        torch_dtype=torch.float32 if use_quantized else torch.float16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager"
    )
    return model, processor

def generate_context(model, processor, image_path, captions, device):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Format the captions with numbering
    formatted_captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])
    
    # Construct the query for image analysis
    query = f"""Analyze the provided image in relation to the following captions:

{formatted_captions}

Generate a concise context (4-5 sentences) that describes the image content, infers its source information, and highlights any relevant information that connects the image to the captions.

Context:"""

    # Define the conversation template for the AI model
    conversation = [
        {
            "role": "assistant", 
            "content": [
                {
                    "type": "text", 
                    "text": "You are an AI assistant skilled in analyzing images and text for fact-checking. Your task is to create a context that helps verify the source and relevance of the image based on the provided captions and your knowledge."
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": query}, 
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # Apply the chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate context
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    
    # Decode and return the generated context
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    print("Generated text: ", generated_text)
    
    # Remove the input prompt from the generated text
    context = generated_text.split("[/INST]")[1].strip()
    
    return context

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
    # model.to(device)
    
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
