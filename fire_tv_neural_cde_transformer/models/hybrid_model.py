# models/hybrid_model.py (Updated for TMDb)
import torch
import torch.nn as nn
from .neural_cde import LayerNormNeuralCDE
from .transformers import BehavioralSequenceTransformer, FireTVRecommendationTransformer

class HybridFireTVSystem(nn.Module):
    """
    TMDb-Enhanced Hybrid Model
    """
    
    def __init__(self, config):
        super().__init__()
        
        print("🔥 EXECUTING TMDb-ENHANCED HYBRID MODEL 🔥")

        
        self.config = config
        self.input_dim = config.input_dim 
        self.output_dim = config.output_dim

        # Core components
        self.neural_cde = LayerNormNeuralCDE(
            input_dim=self.input_dim, 
            hidden_dim=config.neural_cde.hidden_dim,
            dropout_rate=config.neural_cde.dropout_rate,
            vector_field_layers=config.neural_cde.vector_field_layers
        )
        
        self.behavioral_transformer = BehavioralSequenceTransformer(
            feature_dim=self.input_dim,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            patch_size=config.transformer.patch_size
        )
        
        # TMDb feature processors 
        self.tmdb_feature_dim = 70  # Increased from 50 for richer features
        self.tmdb_processor = nn.Sequential(
            nn.Linear(self.tmdb_feature_dim, 256),  # Larger network for richer data
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        self.content_embedding_dim = 384
        self.content_embedding_processor = nn.Sequential(
            nn.Linear(self.content_embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Dynamic fusion layers
        self.cde_fusion = None
        self.transformer_fusion = None
        self.final_fusion = None
        self.layers_initialized = False
        
        # Enhanced prediction heads
        self.content_similarity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.genre_prediction_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 20),  # 20 major genres
            nn.Sigmoid()
        )
        
        self.rating_prediction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _initialize_fusion_layers(self, cde_features, transformer_features):
        """Initialize enhanced fusion layers for TMDb integration"""
        if self.layers_initialized:
            return
            
        cde_dim = cde_features.shape[-1]
        transformer_dim = transformer_features.shape[-1]
        device = cde_features.device
        
        print(f"🔧 Initializing TMDb-enhanced fusion layers:")
        print(f"   CDE features: {cde_dim} dims")
        print(f"   Transformer features: {transformer_dim} dims")
        print(f"   TMDb features: {self.tmdb_feature_dim} dims")
        print(f"   Content embeddings: {self.content_embedding_dim} dims")
        
        # Enhanced processors for richer TMDb data
        self.cde_fusion = nn.Sequential(
            nn.Linear(cde_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        ).to(device)
        
        self.transformer_fusion = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        ).to(device)
        
        # Move TMDb processors to device
        self.tmdb_processor = self.tmdb_processor.to(device)
        self.content_embedding_processor = self.content_embedding_processor.to(device)
        self.content_similarity_head = self.content_similarity_head.to(device)
        self.genre_prediction_head = self.genre_prediction_head.to(device)
        self.rating_prediction_head = self.rating_prediction_head.to(device)
        
        # Enhanced final fusion: 64 * 4 = 256
        self.final_fusion = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.output_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.layers_initialized = True
        print("✅ TMDb-enhanced fusion layers initialized")

    def forward(self, interaction_data, tmdb_features=None, content_embeddings=None, **kwargs):
        """Enhanced forward pass with comprehensive TMDb integration"""
        interaction_path = interaction_data.get('sequence')
        timestamps = interaction_data.get('timestamps')
        
        # Core model processing
        cde_output = self.neural_cde(interaction_path, timestamps)
        cde_features = cde_output[0] if isinstance(cde_output, tuple) else cde_output
        
        transformer_output = self.behavioral_transformer(interaction_path)
        transformer_features = transformer_output[0] if isinstance(transformer_output, tuple) else transformer_output
        
        # Initialize fusion layers
        self._initialize_fusion_layers(cde_features, transformer_features)
        
        # Process core features
        processed_cde = self.cde_fusion(cde_features)
        processed_transformer = self.transformer_fusion(transformer_features)
        
        # Process TMDb features
        if tmdb_features is not None and tmdb_features.shape[0] == cde_features.shape[0]:
            processed_tmdb = self.tmdb_processor(tmdb_features)
        else:
            processed_tmdb = torch.zeros(cde_features.shape[0], 64, device=cde_features.device)
        
        # Process content embeddings
        if content_embeddings is not None and content_embeddings.shape[0] == cde_features.shape[0]:
            processed_content = self.content_embedding_processor(content_embeddings)
        else:
            processed_content = torch.zeros(cde_features.shape[0], 64, device=cde_features.device)
        
        # Combine all features
        combined_features = torch.cat([
            processed_cde, 
            processed_transformer, 
            processed_tmdb, 
            processed_content
        ], dim=-1)
        
        # Generate comprehensive outputs
        final_traits = self.final_fusion(combined_features)
        content_affinity = self.content_similarity_head(processed_content)
        genre_preferences = self.genre_prediction_head(processed_tmdb)
        rating_prediction = self.rating_prediction_head(combined_features)
        
        return {
            'psychological_traits': final_traits,
            'content_affinity_scores': content_affinity,
            'genre_preferences': genre_preferences,
            'predicted_rating': rating_prediction,
            'cde_traits': processed_cde,
            'transformer_traits': processed_transformer,
            'tmdb_traits': processed_tmdb,
            'content_traits': processed_content
        }
