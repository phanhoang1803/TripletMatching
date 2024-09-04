import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from transformers import XLMRobertaModel, XLMRobertaConfig

class TripletMatchingModel(nn.Module):
    def __init__(self, output_dim=768):
        super(TripletMatchingModel, self).__init__()
        self.image_encoder = ImageEncoder(output_dim)
        self.text_encoder = TextEncoder(output_dim)
        
        # Using XLM-RoBERTa for multimodal fusion
        xlm_config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
        xlm_config.num_hidden_layers = 4  # Reduce layers for efficiency
        self.multimodal_transformer = XLMRobertaModel(xlm_config)
        
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, image, caption, context):
        image_features = self.image_encoder(image)
        caption_features = self.text_encoder(caption)
        context_features = self.text_encoder(context)
        
        # Combine features
        combined_features = torch.stack([image_features, caption_features, context_features], dim=1)
        
        # Multi-modal fusion
        fused_features = self.multimodal_transformer(inputs_embeds=combined_features).last_hidden_state[:, 0, :]
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

    def get_features(self):
        # This method should be implemented to return features for the triplet loss
        pass