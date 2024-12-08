import torch
import time
import os
import numpy as np
from thop import profile
import glob
from tqdm import tqdm

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_speed(model, input_size=(1, 3, 32, 32), iterations=100, device='cuda'):
    """Measure inference speed of a model"""
    model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
                
            times.append(time.time() - start)
    
    # Calculate statistics
    times = np.array(times) * 1000  # Convert to milliseconds
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time

def analyze_models(model_dir="./pruned_models", device='cuda'):
    """Analyze all saved models in the directory"""
    
    # Find all saved models
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    results = []
    
    for model_file in tqdm(model_files, desc="Analyzing models"):
        # Load model data
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Extract metadata
        method = checkpoint.get('method', 'unknown')
        sparsity = checkpoint.get('sparsity', 0.0)
        group_idx = checkpoint.get('group_idx', 0)
        accuracy = checkpoint.get('accuracy', 0.0)
        
        # Load model architecture (assuming you're using the same architecture)
        if 'vgg' in model_file.lower():
            from vgg import VGG
            model = VGG('VGG11', num_classes=10)
        elif 'resnet' in model_file.lower():
            from resnet import ResNet18
            model = ResNet18()
        else:
            print(f"Unknown model architecture for {model_file}")
            continue
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Count parameters
        num_params = count_parameters(model)
        
        # Measure inference speed
        mean_time, std_time = measure_inference_speed(model, device=device)
        
        # Calculate FLOPs
        dummy_input = torch.randn(1, 3, 32, 32)
        macs, _ = profile(model, inputs=(dummy_input,))
        
        results.append({
            'file': model_file,
            'method': method,
            'sparsity': sparsity,
            'group_idx': group_idx,
            'accuracy': accuracy,
            'parameters': num_params,
            'inference_time_ms': mean_time,
            'inference_time_std_ms': std_time,
            'macs': macs
        })
    
    return results

def print_comparison(results):
    """Print comparison table of results"""
    
    # Group results by sparsity and group_idx
    grouped_results = {}
    for result in results:
        key = (result['sparsity'], result['group_idx'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 100)
    print(f"{'Sparsity':10} {'Group':8} {'Method':15} {'Accuracy':10} {'Parameters':12} {'Inference(ms)':15} {'MACs':12}")
    print("-" * 100)
    
    for key in sorted(grouped_results.keys()):
        sparsity, group_idx = key
        results = grouped_results[key]
        
        for result in results:
            print(f"{result['sparsity']:<10.2f} "
                  f"{result['group_idx']:<8d} "
                  f"{result['method']:<15s} "
                  f"{result['accuracy']:<10.2f} "
                  f"{result['parameters']:<12,d} "
                  f"{result['inference_time_ms']:<6.2f}±{result['inference_time_std_ms']:<6.2f} "
                  f"{result['macs']/1e6:<12.2f}M")
        print("-" * 100)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./pruned_models',
                        help='Directory containing saved models')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for speed testing')
    
    args = parser.parse_args()
    
    # Analyze all models
    results = analyze_models(args.model_dir, device=args.device)
    
    # Print comparison
    print_comparison(results)