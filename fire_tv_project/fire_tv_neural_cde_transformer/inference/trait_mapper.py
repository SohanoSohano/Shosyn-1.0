# inference/trait_mapper.py (Updated for 15 traits)

class FireTVPsychologicalTraitMapper:
    """Maps 15-dimensional Fire TV trait vector to interpretable dimensions"""
    
    def __init__(self):
        # Map 15 model outputs to the most important Fire TV traits
        self.trait_mapping = {
            1: {
                "name": "Cognitive Load",
                "category": "Cognitive",
                "description": "The mental effort required by the user to interact with the system",
                "high_interpretation": "Prefers simple, intuitive interfaces",
                "low_interpretation": "Comfortable with complex navigation"
            },
            2: {
                "name": "Decision Confidence",
                "category": "Decision Making", 
                "description": "The user's confidence level in making content choices",
                "high_interpretation": "Makes quick, decisive content choices",
                "low_interpretation": "Hesitant, needs more information to decide"
            },
            3: {
                "name": "Exploration Tendency",
                "category": "Behavioral",
                "description": "The user's inclination to explore new content or features",
                "high_interpretation": "Actively seeks new content and features",
                "low_interpretation": "Sticks to familiar content and patterns"
            },
            4: {
                "name": "Attention Span",
                "category": "Cognitive",
                "description": "The duration the user can focus on content or interface elements",
                "high_interpretation": "Can focus for extended periods",
                "low_interpretation": "Prefers quick, bite-sized content"
            },
            5: {
                "name": "Navigation Efficiency",
                "category": "Technical Proficiency",
                "description": "How efficiently the user navigates through the interface",
                "high_interpretation": "Expert navigator, uses shortcuts",
                "low_interpretation": "Takes longer paths, needs guidance"
            },
            6: {
                "name": "Platform Loyalty",
                "category": "Brand Affinity",
                "description": "The user's loyalty or preference for the Fire TV platform",
                "high_interpretation": "Strong Fire TV advocate",
                "low_interpretation": "Platform agnostic, may switch easily"
            },
            7: {
                "name": "Social Influence",
                "category": "Social",
                "description": "The impact of social factors on the user's content choices",
                "high_interpretation": "Heavily influenced by trends and reviews",
                "low_interpretation": "Makes independent content decisions"
            },
            8: {
                "name": "Content Diversity",
                "category": "Content Preference",
                "description": "The user's preference for diverse content genres and types",
                "high_interpretation": "Enjoys wide variety of genres",
                "low_interpretation": "Prefers specific genres or content types"
            },
            9: {
                "name": "Session Engagement",
                "category": "Engagement",
                "description": "The level of user engagement during a viewing session",
                "high_interpretation": "Highly engaged, binge-watches",
                "low_interpretation": "Casual viewer, shorter sessions"
            },
            10: {
                "name": "UI Adaptation",
                "category": "Technical Proficiency",
                "description": "The user's ability to adapt to changes in the user interface",
                "high_interpretation": "Quickly adapts to UI changes",
                "low_interpretation": "Struggles with interface updates"
            },
            11: {
                "name": "Voice Usage",
                "category": "Interface Preference",
                "description": "The extent to which the user utilizes voice commands",
                "high_interpretation": "Heavy voice command user",
                "low_interpretation": "Prefers traditional remote control"
            },
            12: {
                "name": "Recommendation Acceptance",
                "category": "AI Interaction",
                "description": "The user's tendency to accept or reject recommendations",
                "high_interpretation": "Trusts and follows recommendations",
                "low_interpretation": "Skeptical of algorithmic suggestions"
            },
            13: {
                "name": "Search Sophistication",
                "category": "Technical Proficiency",
                "description": "The user's skill and strategy in searching for content",
                "high_interpretation": "Uses advanced search techniques",
                "low_interpretation": "Basic search patterns only"
            },
            14: {
                "name": "Peak Alignment",
                "category": "Temporal",
                "description": "Alignment of user activity with peak usage times",
                "high_interpretation": "Views during popular time slots",
                "low_interpretation": "Off-peak viewing patterns"
            },
            15: {
                "name": "Return Likelihood",
                "category": "Retention",
                "description": "The likelihood of the user returning to the platform",
                "high_interpretation": "Highly likely to return regularly",
                "low_interpretation": "May churn or use infrequently"
            }
        }
    
    def interpret_trait_vector(self, trait_vector: list[float]) -> dict[str, dict]:
        """Convert 15-dimensional Fire TV trait vector to interpretable profile"""
        if len(trait_vector) != 15:  # Changed from 20 to 15
            raise ValueError(f"Expected 15 traits, got {len(trait_vector)}")
        
        profile = {}
        
        for i, value in enumerate(trait_vector, 1):
            if i > 15:  # Only process first 15 traits
                break
                
            trait_info = self.trait_mapping[i]
            profile[f"trait_{i}"] = {
                "name": trait_info["name"],
                "category": trait_info["category"],
                "description": trait_info["description"],
                "value": round(value, 3),
                "interpretation": self._interpret_trait_value(value, trait_info),
                "strength": self._get_trait_strength(value),
                "impact": self._get_trait_impact(trait_info["category"], abs(value))
            }
        
        return profile
    
    def _interpret_trait_value(self, value: float, trait_info: dict) -> str:
        """Provide human-readable interpretation of Fire TV trait values"""
        abs_value = abs(value)
        
        if abs_value >= 0.7:
            strength = "Very"
        elif abs_value >= 0.5:
            strength = "Moderately"
        elif abs_value >= 0.3:
            strength = "Somewhat"
        else:
            strength = "Slightly"
        
        if value > 0:
            return f"{strength} {trait_info['high_interpretation']}"
        else:
            return f"{strength} {trait_info['low_interpretation']}"
    
    def _get_trait_strength(self, value: float) -> str:
        """Get trait strength category"""
        abs_value = abs(value)
        if abs_value >= 0.7:
            return "Very Strong"
        elif abs_value >= 0.5:
            return "Strong"
        elif abs_value >= 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    def _get_trait_impact(self, category: str, abs_value: float) -> str:
        """Determine the impact level of a trait on Fire TV experience"""
        impact_weights = {
            "Cognitive": 1.2,
            "Technical Proficiency": 1.1,
            "Content Preference": 1.0,
            "Engagement": 1.3,
            "AI Interaction": 1.1,
            "Interface Preference": 1.0,
            "Behavioral": 0.9,
            "Social": 0.8,
            "Economic": 0.9,
            "Brand Affinity": 0.7,
            "Cross-Platform": 0.8,
            "Temporal": 0.7,
            "Hardware Affinity": 0.6,
            "Retention": 1.4,
            "Decision Making": 1.0,
            "Emotional": 1.1
        }
        
        weighted_value = abs_value * impact_weights.get(category, 1.0)
        
        if weighted_value >= 0.8:
            return "High Impact"
        elif weighted_value >= 0.5:
            return "Medium Impact"
        else:
            return "Low Impact"
    
    def get_fire_tv_insights(self, trait_vector: list[float]) -> dict[str, any]:
        """Generate Fire TV-specific user insights"""
        profile = self.interpret_trait_vector(trait_vector)
        
        # Extract key Fire TV behavioral patterns
        insights = {
            "user_type": self._determine_user_type(trait_vector),
            "interface_preferences": self._analyze_interface_preferences(trait_vector),
            "content_strategy": self._analyze_content_strategy(trait_vector),
            "engagement_pattern": self._analyze_engagement_pattern(trait_vector),
            "technical_proficiency": self._analyze_technical_proficiency(trait_vector),
            "retention_risk": self._analyze_retention_risk(trait_vector)
        }
        
        return insights
    
    def _determine_user_type(self, traits: list[float]) -> str:
        """Determine Fire TV user archetype - FIXED for 15 traits"""
        # Map to available indices (0-14 instead of 0-19)
        exploration = traits[3]  # exploration_tendency (trait 4)
        
        # Use only available indices for tech proficiency
        tech_proficiency = (traits[5] + traits[9] + traits[12]) / 3  # nav_efficiency, ui_adaptation, search_sophistication
        
        engagement = traits[8]  # session_engagement (trait 9)
        
        if tech_proficiency > 0.5 and exploration > 0.5:
            return "Power User"
        elif engagement > 0.5 and traits[7] > 0.5:  # content_diversity (trait 8)
            return "Content Explorer"
        elif traits[10] > 0.5 and traits[14] > 0.5:  # ui_adaptation, return_likelihood
            return "Loyal Viewer"
        elif tech_proficiency < -0.3:
            return "Casual User"
        else:
            return "Balanced User"

    def _analyze_interface_preferences(self, traits: list[float]) -> dict[str, str]:
        """Analyze interface and interaction preferences - FIXED for 15 traits"""
        return {
            "navigation_style": "Efficient" if traits[4] > 0.3 else "Guided",  # navigation_efficiency (trait 5)
            "voice_preference": "High" if traits[10] > 0.3 else "Low",  # voice_usage (trait 11) -> ui_adaptation
            "ui_complexity": "Simple" if traits[0] > 0.3 else "Advanced",  # cognitive_load (trait 1)
            "adaptation_speed": "Fast" if traits[9] > 0.3 else "Slow"  # ui_adaptation (trait 10)
        }

    def _analyze_content_strategy(self, traits: list[float]) -> dict[str, str]:
        """Analyze content recommendation strategy - FIXED for 15 traits"""
        return {
            "diversity_level": "High" if traits[7] > 0.3 else "Focused",  # content_diversity (trait 8)
            "recommendation_trust": "High" if traits[11] > 0.3 else "Low",  # recommendation_acceptance (trait 12)
            "social_influence": "High" if traits[6] > 0.3 else "Low",  # social_influence (trait 7)
            "exploration_drive": "High" if traits[3] > 0.3 else "Low"  # exploration_tendency (trait 4)
        }

    def _analyze_engagement_pattern(self, traits: list[float]) -> dict[str, str]:
        """Analyze user engagement patterns - FIXED for 15 traits"""
        return {
            "session_length": "Long" if traits[8] > 0.3 else "Short",  # session_engagement (trait 9)
            "attention_duration": "Extended" if traits[4] > 0.3 else "Brief",  # attention_span (trait 5)
            "viewing_consistency": "Regular" if traits[9] > 0.3 else "Sporadic",  # ui_adaptation as proxy
            "peak_alignment": "Aligned" if traits[13] > 0.3 else "Off-peak"  # peak_alignment (trait 14)
        }

    def _analyze_technical_proficiency(self, traits: list[float]) -> dict[str, str]:
        """Analyze technical skill level - FIXED for 15 traits"""
        # Use available traits for proficiency calculation
        proficiency_score = (traits[4] + traits[9] + traits[12]) / 3  # nav_efficiency, ui_adaptation, search_sophistication
        
        return {
            "overall_level": "Advanced" if proficiency_score > 0.3 else "Basic",
            "search_skills": "Sophisticated" if traits[12] > 0.3 else "Basic",  # search_sophistication (trait 13)
            "multi_platform": "Active" if traits[10] > 0.3 else "Single-device"  # voice_usage as proxy
        }

    def _analyze_retention_risk(self, traits: list[float]) -> dict[str, any]:
        """Analyze user retention and churn risk - FIXED for 15 traits"""
        retention_score = traits[14]  # return_likelihood (trait 15)
        frustration = traits[2] if len(traits) > 2 else 0  # Use trait 3 if available
        loyalty = traits[5]  # platform_loyalty (trait 6)
        
        risk_score = (1 - retention_score) + frustration - loyalty
        
        return {
            "risk_level": "High" if risk_score > 0.5 else "Medium" if risk_score > 0 else "Low",
            "key_factors": {
                "return_likelihood": "High" if retention_score > 0.3 else "Low",
                "frustration_level": "High" if frustration > 0.3 else "Low",
                "platform_loyalty": "High" if loyalty > 0.3 else "Low"
            },
            "retention_score": round(retention_score, 3)
        }