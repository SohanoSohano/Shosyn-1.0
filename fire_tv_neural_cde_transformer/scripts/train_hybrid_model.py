# scripts/train_hybrid_model.py
import sys
import os
import torch
import torch.cuda
import gc
import wandb
from torch.utils.data import DataLoader

# Set CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.model_config import HybridModelConfig
from config.training_config import TrainingConfig
from data.data_loader import create_data_loaders
from models.hybrid_model import HybridFireTVSystem
from training.trainer import HybridModelTrainer

def clear_gpu_memory():
    """Clear GPU memory and check availability"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Test GPU with small tensor
            test_tensor = torch.randn(10, 10).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            print("✅ GPU memory cleared and available")
            return True
        except RuntimeError as e:
            print(f"❌ GPU test failed: {e}")
            return False
    return False

def get_safe_device():
    """Safely detect and return available device"""
    
    # Clear GPU memory first
    gpu_available = clear_gpu_memory()
    
    if gpu_available:
        try:
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        except Exception as e:
            print(f"❌ GPU initialization failed: {e}")
    
    print("💻 Using CPU for training")
    return torch.device('cpu')

def debug_data_loader(data_loader):
    """Debug function to check data loader output"""
    print("🔍 Debugging data loader...")
    
    try:
        for i, batch in enumerate(data_loader):
            print(f"Batch {i}:")
            print(f"  Type: {type(batch)}")
            
            if isinstance(batch, dict):
                print(f"  Keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"    {key}: dict with keys {list(value.keys())}")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                print(f"      {subkey}: {subvalue.shape}")
                    else:
                        print(f"    {key}: {type(value)} - length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            else:
                print(f"  Content type: {type(batch)}")
            
            if i >= 1:  # Only check first 2 batches
                break
                
    except Exception as e:
        print(f"❌ Error in debug: {e}")

def main():
    """FIXED: Main training script with proper variable scope"""
    
    print("🧪 Fire TV Neural CDE + Transformer Training")
    print("=" * 50)
    
    # Initialize variables to None
    train_loader = None
    val_loader = None
    model = None
    trainer = None
    
    try:
        # Get safe device
        device = get_safe_device()
        
        # Initialize wandb
        wandb.init(project="fire-tv-neural-cde-transformer")
        
        # Configuration with reduced model size for GPU compatibility
        model_config = HybridModelConfig()
        training_config = TrainingConfig()
        
        # Adjust batch size based on device and GPU memory
        if device.type == 'cuda':
            # For RTX 3050 with 4.3GB, use smaller batch size
            training_config.batch_size = 4
            print("📉 Reduced batch size to 4 for GPU memory optimization")
        else:
            training_config.batch_size = 2
            print("📉 Reduced batch size to 2 for CPU training")
        
        # Data loading with error handling
        print("📊 Loading data...")
        try:
            train_loader, val_loader = create_data_loaders(
                data_path="data/fire_tv_dataset.csv",
                batch_size=training_config.batch_size,
                validation_split=training_config.validation_split
            )
            print("✅ Data loaded successfully")
            
            # Debug data structure AFTER successful loading
            print("🔍 Debugging data structure...")
            debug_data_loader(train_loader)
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            print("🔄 Trying with simple data loader...")
            
            # Fallback to simple data loader
            from data.simple_data_loader import create_simple_data_loaders
            train_loader, val_loader = create_simple_data_loaders(
                batch_size=training_config.batch_size,
                num_samples=800
            )
            print("✅ Simple data loader created successfully")
            debug_data_loader(train_loader)
        
        # Model creation
        print("🏗️ Creating hybrid model...")
        try:
            model = HybridFireTVSystem(model_config)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ Model created with {param_count:,} parameters")
            
            # Memory estimation
            if device.type == 'cuda':
                model_memory = param_count * 4 / 1e9  # 4 bytes per float32 parameter
                print(f"📊 Estimated model memory: {model_memory:.2f} GB")
                
                # Check if model fits in GPU memory
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if model_memory > available_memory * 0.8:  # Use 80% of available memory
                    print(f"⚠️ Model may be too large for GPU ({model_memory:.2f}GB > {available_memory*0.8:.2f}GB)")
                    print("🔄 Switching to CPU...")
                    device = torch.device('cpu')
                
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return None
        
        # Trainer initialization with error handling
        print("🎯 Initializing trainer...")
        try:
            trainer = HybridModelTrainer(model, training_config, device)
            print("✅ Trainer initialized successfully")
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error) or "out of memory" in str(cuda_error).lower():
                print(f"❌ GPU error in trainer: {cuda_error}")
                print("🔄 Retrying with CPU...")
                device = torch.device('cpu')
                trainer = HybridModelTrainer(model, training_config, device)
                print("✅ Trainer initialized with CPU")
            else:
                print(f"❌ Trainer initialization failed: {cuda_error}")
                return None
        
        # Training
        print("🚀 Starting training...")
        try:
            history = trainer.train(train_loader, val_loader, training_config.num_epochs)
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Save final model
        try:
            model_path = f"models/final_hybrid_model_{device.type}.pth"
            os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved as {model_path}")
        except Exception as e:
            print(f"❌ Model saving failed: {e}")
        
        return history
        
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up resources
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    try:
        history = main()
        if history:
            print("🎉 Training pipeline completed successfully!")
        else:
            print("❌ Training pipeline failed")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            wandb.finish()
        except:
            pass
