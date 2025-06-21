import sys
sys.path.append('/home/ubuntu/fire_tv_project/fire_tv_neural_cde_transformer')

from data.dummy_tmdb_integration import DummyTMDbIntegration
import torch

def test_dummy_integration():
    print("🧪 Testing Dummy TMDb Integration...")
    
    # Initialize dummy integration
    dummy_tmdb = DummyTMDbIntegration('cuda')
    
    # Test with sample content IDs
    test_content_ids = ["content_0", "content_1", "content_2", "content_100"]
    
    # Test feature generation
    tmdb_features = dummy_tmdb.create_tmdb_features(test_content_ids)
    content_embeddings = dummy_tmdb.create_content_embeddings(test_content_ids)
    tmdb_data = dummy_tmdb.fetch_tmdb_data(test_content_ids, {})
    
    print(f"✅ TMDb Features: {tmdb_features.shape}")
    print(f"✅ Content Embeddings: {content_embeddings.shape}")
    print(f"✅ TMDb Data: {len(tmdb_data)} items")
    
    # Test consistency (same input should give same output)
    features_1 = dummy_tmdb.create_tmdb_features(["content_0"])
    features_2 = dummy_tmdb.create_tmdb_features(["content_0"])
    
    if torch.allclose(features_1, features_2):
        print("✅ Consistency test passed")
    else:
        print("❌ Consistency test failed")
    
    print("🎯 Dummy TMDb integration ready!")

if __name__ == "__main__":
    test_dummy_integration()
