# models/transformers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import PositionalEncoding, PathwayAttention, MultiModalFusion, PsychologicalTraitDecoder

class BehavioralSequenceTransformer(nn.Module):
    """Transformer for modeling Fire TV behavioral sequences"""
    
    def __init__(self, feature_dim=49, d_model=512, nhead=8, num_layers=6, patch_size=4):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding for behavioral sequences
        self.patch_embedding = nn.Linear(feature_dim * patch_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pathway attention for behavior focus
        self.pathway_attention = PathwayAttention(d_model, nhead)
        
        # Psychological trait decoder
        self.trait_decoder = PsychologicalTraitDecoder(d_model)
        
    def forward(self, behavioral_sequence):
        batch_size, seq_len, feature_dim = behavioral_sequence.shape
        
        # Create patches from behavioral sequence
        num_patches = seq_len // self.patch_size
        if num_patches == 0:
            # Handle short sequences
            padded_sequence = F.pad(
                behavioral_sequence, 
                (0, 0, 0, self.patch_size - seq_len)
            )
            num_patches = 1
            patches = padded_sequence.reshape(
                batch_size, num_patches, feature_dim * self.patch_size
            )
        else:
            patches = behavioral_sequence[:, :num_patches * self.patch_size].reshape(
                batch_size, num_patches, feature_dim * self.patch_size
            )
        
        # Embed patches
        embedded_patches = self.patch_embedding(patches)
        embedded_patches = self.pos_encoding(embedded_patches)
        
        # Apply transformer
        transformer_output = self.transformer(embedded_patches)
        
        # Apply pathway attention
        pathway_output, attention_weights = self.pathway_attention(transformer_output)
        
        # Global average pooling
        pooled_output = pathway_output.mean(dim=1)
        
        # Predict psychological traits
        trait_predictions = self.trait_decoder(pooled_output)
        
        return trait_predictions, attention_weights

class MultiModalFireTVTransformer(nn.Module):
    """Multi-modal transformer for different Fire TV data types"""
    
    def __init__(self, input_dims, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Modal-specific encoders
        self.modal_encoders = nn.ModuleDict({
            modal: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for modal, dim in input_dims.items()
        })
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            {modal: hidden_dim for modal in input_dims.keys()}, 
            hidden_dim
        )
        
        # Trait decoder
        self.trait_decoder = PsychologicalTraitDecoder(hidden_dim)
        
    def forward(self, modal_inputs):
        # Encode each modality
        encoded_modals = {}
        for modal, data in modal_inputs.items():
            if modal in self.modal_encoders:
                encoded = self.modal_encoders[modal](data)
                encoded_modals[modal] = encoded
        
        # Stack modalities for transformer processing
        modal_sequence = torch.stack(list(encoded_modals.values()), dim=1)
        modal_sequence = self.pos_encoding(modal_sequence)
        
        # Apply transformer
        transformed_features = self.transformer(modal_sequence)
        
        # Fuse modalities
        fused_features = self.fusion(
            {modal: transformed_features[:, i] 
             for i, modal in enumerate(encoded_modals.keys())}
        )
        
        # Predict traits
        trait_predictions = self.trait_decoder(fused_features)
        
        return trait_predictions

class FireTVRecommendationTransformer(nn.Module):
    """Transformer-based recommender with psychological trait integration"""
    
    def __init__(self, content_vocab_size, psychological_traits=20, 
                 d_model=512, max_seq_len=100):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Content embeddings
        self.content_embedding = nn.Embedding(content_vocab_size, d_model)
        
        # Psychological trait integration
        self.trait_projection = nn.Linear(psychological_traits, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Pathway attention for behavior-aware recommendations
        self.pathway_attention = PathwayAttention(d_model)
        
        # Transformer decoder for content generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, content_vocab_size)
        
    def forward(self, user_history, psychological_traits, target_sequence=None):
        batch_size = user_history.size(0)
        
        # Embed user history
        history_embedded = self.content_embedding(user_history)
        history_embedded = self.pos_encoding(history_embedded)
        
        # Integrate psychological traits
        trait_features = self.trait_projection(psychological_traits)
        trait_features = trait_features.unsqueeze(1).expand(-1, history_embedded.size(1), -1)
        
        # Combine content and psychological features
        combined_features = history_embedded + trait_features
        
        # Apply pathway attention
        pathway_features, attention_weights = self.pathway_attention(combined_features)
        
        if target_sequence is not None:
            # Training mode
            target_embedded = self.content_embedding(target_sequence)
            target_embedded = self.pos_encoding(target_embedded)
            
            # Create causal mask
            seq_len = target_embedded.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            
            # Apply transformer decoder
            output = self.transformer_decoder(
                target_embedded, pathway_features, tgt_mask=causal_mask
            )
        else:
            # Inference mode - autoregressive generation
            output = self.generate_recommendations(pathway_features, max_length=10)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits, attention_weights
    
    def generate_recommendations(self, memory, max_length=10):
        """Autoregressive recommendation generation"""
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with a special token (e.g., 0 for <START>)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Embed current sequence
            embedded = self.content_embedding(generated)
            embedded = self.pos_encoding(embedded)
            
            # Apply transformer decoder
            output = self.transformer_decoder(embedded, memory)
            
            # Get next token logits
            next_token_logits = self.output_projection(output[:, -1:])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return self.content_embedding(generated)
