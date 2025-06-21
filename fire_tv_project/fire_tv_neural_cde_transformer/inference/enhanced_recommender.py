# inference/enhanced_recommender.py

import torch
import numpy as np
from typing import List, Dict

# --- MODIFICATION START ---
# Import the new data enricher and os to get the API key
from recommender import RecommendationService
from score_interpreter import ScoreInterpreter
from trait_mapper import FireTVPsychologicalTraitMapper
from confidence_calculator import ConfidenceCalculator
from data_enricher import TMDbDataEnricher  
import os
# --- MODIFICATION END ---


class EnhancedFireTVRecommendationService(RecommendationService):
    """Enhanced Fire TV recommendation service with real movie data enrichment"""
    
    def __init__(self, model_path: str, item_catalog_path: str, device: str = 'cpu'):
        super().__init__(model_path, item_catalog_path, device)
        
        # Initialize Fire TV-specific enhancement components
        self.score_interpreter = ScoreInterpreter()
        self.trait_mapper = FireTVPsychologicalTraitMapper()
        self.confidence_calculator = ConfidenceCalculator()

        # --- MODIFICATION START ---
        # Initialize the TMDbDataEnricher with your API key
        # It's best practice to load secrets from environment variables
        tmdb_api_key = os.getenv("c799fe85bcebb074eff49aa01dc6cdb0", "c799fe85bcebb074eff49aa01dc6cdb0") # Replace with your key if not using env vars
        self.enricher = TMDbDataEnricher(api_key=tmdb_api_key)
        # --- MODIFICATION END ---
    
    def get_enhanced_recommendations(self, 
                                   user_id: str, 
                                   user_history: List[Dict], 
                                   top_k: int = 10) -> Dict:
        """Get Fire TV recommendations enriched with real movie names and platform data"""
        
        # Get base recommendations
        base_recommendations = self.get_recommendations(user_id, user_history, top_k)
        
        # Extract 15-dimensional user traits
        processed_history = self._preprocess_user_history(user_history)
        timestamps = torch.arange(processed_history.shape[1]).float().unsqueeze(0).to(self.device)
        interaction_data = {'sequence': processed_history, 'timestamps': timestamps}
        dummy_tmdb_features = torch.randn(1, 70).to(self.device)
        dummy_embeddings = torch.randn(1, 384).to(self.device)

        with torch.inference_mode():
            model_output = self.model(
                interaction_data,
                tmdb_features=dummy_tmdb_features,
                content_embeddings=dummy_embeddings
            )
            user_trait_vector = model_output['psychological_traits'].cpu().numpy().flatten()
        
        actual_trait_count = len(user_trait_vector)
        print(f"🔍 Processing {actual_trait_count} psychological traits")
        
        if len(user_trait_vector) != 15:
            raise ValueError(f"Model should output 15 traits, got {len(user_trait_vector)}")
        
        fire_tv_insights = self.trait_mapper.get_fire_tv_insights(user_trait_vector.tolist())
        user_profile = self.trait_mapper.interpret_trait_vector(user_trait_vector.tolist())
        
        # Enhanced recommendations with Fire TV context
        enhanced_recommendations = []
        
        for i, recommendation in enumerate(base_recommendations):
            score_info = self.score_interpreter.interpret_score(recommendation['score'])
            
            # --- MODIFICATION START ---
            # Enrich each recommendation with TMDb data
            enriched_data = self.enricher.get_movie_details(recommendation['item_id'])
            if enriched_data is None:
                # If enrichment fails, use original data and mark it
                enriched_data = {
                    "title": recommendation['title'], # Fallback to original title
                    "overview": "Details not available.",
                    "streaming_on": ["N/A"]
                }
            # --- MODIFICATION END ---

            named_traits = {}
            for trait_num in range(1, 16):
                trait_key = f"trait_{trait_num}"
                if trait_key in recommendation:
                    trait_name = self.trait_mapper.trait_mapping[trait_num]['name']
                    named_traits[trait_name] = recommendation[trait_key]

            enhanced_rec = {
                #"item_id": recommendation['item_id'],
                "enriched_data": enriched_data, # Add the new dictionary with TMDb data
                "score_interpretation": score_info,
                "named_psychological_traits": named_traits,
                "fire_tv_relevance": self._calculate_fire_tv_relevance(
                    recommendation, fire_tv_insights
                ),
                # You can optionally remove the numbered traits if you only want named ones
                **recommendation, # This includes the original trait_1, trait_2, etc.
            }
            enhanced_recommendations.append(enhanced_rec)
        
        return {
            "user_id": user_id,
            "fire_tv_profile": {
                "user_type": fire_tv_insights["user_type"],
                "psychological_traits": user_profile,
                "interface_preferences": fire_tv_insights["interface_preferences"],
                "content_strategy": fire_tv_insights["content_strategy"],
                "engagement_pattern": fire_tv_insights["engagement_pattern"],
                "technical_proficiency": fire_tv_insights["technical_proficiency"],
                "retention_analysis": fire_tv_insights["retention_risk"]
            },
            "recommendations": enhanced_recommendations,
            "personalization_summary": self._generate_fire_tv_summary(fire_tv_insights)
        }
    
    def _calculate_fire_tv_relevance(self, recommendation: Dict, insights: Dict) -> Dict:
        """Calculate how well a recommendation fits Fire TV user profile"""
        relevance_factors = []
        if insights["content_strategy"]["diversity_level"] == "High":
            relevance_factors.append("Matches your diverse content preferences")
        user_type = insights["user_type"]
        if user_type == "Power User":
            relevance_factors.append("Suitable for advanced Fire TV users")
        elif user_type == "Content Explorer":
            relevance_factors.append("Great for content discovery")
        return { "relevance_score": len(relevance_factors) / 3, "relevance_factors": relevance_factors }
    
    def _generate_fire_tv_summary(self, insights: Dict) -> str:
        """Generate Fire TV-specific personalization summary"""
        user_type = insights["user_type"]
        interface_pref = insights["interface_preferences"]["navigation_style"]
        content_strategy = insights["content_strategy"]["diversity_level"]
        summary = f"Fire TV Profile: {user_type} with {interface_pref.lower()} navigation preferences. "
        summary += f"Recommending {content_strategy.lower()} content diversity based on your viewing patterns."
        return summary
