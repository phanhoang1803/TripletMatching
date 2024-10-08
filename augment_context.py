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
    
    # Remove duplicate captions
    unique_captions = list(dict.fromkeys(captions))
    
    # Format the unique captions with numbering
    formatted_captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(unique_captions)])

    # Construct the query for image analysis
    query = f"""
Captions:
{formatted_captions}

Given the the image and the captions associated with it, you need to write a detailed context paragraph based on the image and the captions. There are some key points you should cover to generate the context:

1. Describes the key visual elements and content of the image.
2. Infers the most likely context, setting, or event depicted from the captions. Incorporate relevant details from the captions to enhance the description.
3. Estimates the time period and possible geographic location, if possible.

You have to write the context in a single paragraph of 7-10 sentences. Be specific and confident in your description and inferences. Do not mention limitations in identifying specific individuals or places.

Context:"""

    # Define the conversation template for the AI model
    conversation = [
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "You are an AI trained in detailed image and caption analysis. Use both visual cues from the image and information from the captions to generate a coherent and informative context paragraph. Integrate details from the captions seamlessly into your analysis."
            }
        ]
    },
    {
        "role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": query} 
            ]
        }
    ]
    
    # Apply the chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate context
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.8)
    
    # Decode and return the generated context
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    # print("Generated text: ", generated_text)
    
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
