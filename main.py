import copy
import torchvision.transforms as transforms
from utils import evaluate, find_ignored_layers, get_cifar100_data_loader, get_cifar10_data_loader, get_pretrained_model
from prune import prune
from torch_pruning.optimal_transport import OptimalTransport
import torch_pruning as tp
import torch
import json
from parameters import get_parser
import os
from datetime import datetime
import time
import numpy as np

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, loaders, gpu_id, num_iterations=100):
    """Measure average inference time over multiple iterations"""
    if gpu_id != -1:
        model = model.cuda(gpu_id)
    model.eval()
    
    # Get a batch of data
    images, _ = next(iter(loaders["test"]))
    if gpu_id != -1:
        images = images.cuda(gpu_id)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(images)
    
    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Remove outliers (optional)
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    filtered_times = times[abs(times - mean_time) < 2 * std_time]
    
    avg_time = np.mean(filtered_times)
    return avg_time * 1000  # Convert to milliseconds

def save_model(model, accuracy, sparsity, group_idx, method_name, params_info, timing_info, save_dir="./pruned_models"):
    """Save the pruned model along with its metrics"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/pruned_model_{method_name}_group{group_idx}_sparsity{sparsity:.2f}_{timestamp}.pth"
    
    state = {
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'sparsity': sparsity,
        'group_idx': group_idx,
        'method': method_name,
        'parameters': params_info,
        'inference_time': timing_info,
        'timestamp': timestamp
    }
    
    torch.save(state, filename)
    print(f"Model saved to {filename}")
    return filename

def log_results(log_file, group_idx, sparsity, original_metrics, default_metrics, if_metrics):
    """Log results to a JSON file"""
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'group_idx': group_idx,
        'sparsity': sparsity,
        'original_model': original_metrics,
        'default_pruning': default_metrics,
        'intra_fusion': if_metrics
    }
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    log_data.append(results)
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

if __name__ == '__main__':
    # Parse command line arguments
    args = get_parser().parse_args()

    # Set up configuration
    dataset = "Cifar10"
    example_inputs = torch.randn(1, 3, 32, 32)
    out_features = 10
    backward_pruning = True
    file_name = f"./models/{args.model_name}.checkpoint"

    config = dict(
        dataset=dataset,
        model=args.model_name,
    )

    # Load data and model
    loaders = get_cifar10_data_loader()
    model_original, base_accuracy = get_pretrained_model(config, file_name)
        
    # Measure original model metrics
    original_params = count_parameters(model_original)
    original_inference_time = measure_inference_time(model_original, loaders, args.gpu_id)
    
    print("\nOriginal model metrics:")
    print(f"Parameters: {original_params:,}")
    print(f"Base accuracy: {base_accuracy:.2f}%")
    print(f"Inference time: {original_inference_time:.2f}ms")

    # Initialize Optimal Transport
    ot = OptimalTransport(
        p=args.p,
        target_probability=args.target_prob,
        source_probability=args.source_prob,
        target=args.target,
        gpu_id=args.gpu_id
    )

    # Create log file path
    log_file = f"./logs/pruning_results_{datetime.now().strftime('%Y%m%d')}.json"

    # Main pruning loop
    for group_idx in args.group_idxs:
        for sparsity in args.sparsities:
            print(f"\nProcessing group {group_idx} with sparsity {sparsity}")
            
            # Default pruning
            pruned_model_default = copy.deepcopy(model_original)
            prune(
                pruned_model_default,
                loaders,
                example_inputs,
                out_features,
                args.importance_criteria,
                args.gpu_id,
                sparsity=sparsity,
                backward_pruning=backward_pruning,
                group_idxs=[group_idx],
                dimensionality_preserving=False
            )
            
            # Measure default pruning metrics
            accuracy_pruned_default = evaluate(pruned_model_default, loaders, gpu_id=args.gpu_id)
            default_params = count_parameters(pruned_model_default)
            default_inference_time = measure_inference_time(pruned_model_default, loaders, args.gpu_id)

            # Intra-Fusion pruning
            pruned_model_IF = copy.deepcopy(model_original)
            prune(
                pruned_model_IF,
                loaders,
                example_inputs,
                out_features,
                args.importance_criteria,
                args.gpu_id,
                sparsity=sparsity,
                optimal_transport=ot,
                backward_pruning=backward_pruning,
                group_idxs=[group_idx],
                dimensionality_preserving=False
            )
            
            # Measure IF pruning metrics
            accuracy_pruned_IF = evaluate(pruned_model_IF, loaders, gpu_id=args.gpu_id)
            if_params = count_parameters(pruned_model_IF)
            if_inference_time = measure_inference_time(pruned_model_IF, loaders, args.gpu_id)
            
            # Save models and their metrics
            default_params_info = {
                'total': default_params,
                'reduction': original_params - default_params,
                'reduction_percentage': (1 - default_params/original_params) * 100
            }
            
            if_params_info = {
                'total': if_params,
                'reduction': original_params - if_params,
                'reduction_percentage': (1 - if_params/original_params) * 100
            }
            
            default_timing_info = {
                'time': default_inference_time,
                'improvement': (original_inference_time - default_inference_time)/original_inference_time * 100
            }
            
            if_timing_info = {
                'time': if_inference_time,
                'improvement': (original_inference_time - if_inference_time)/original_inference_time * 100
            }
            
            # Save models
            save_model(
                pruned_model_default,
                accuracy_pruned_default,
                sparsity,
                group_idx,
                'default_pruning',
                default_params_info,
                default_timing_info
            )
            
            save_model(
                pruned_model_IF,
                accuracy_pruned_IF,
                sparsity,
                group_idx,
                'intra_fusion',
                if_params_info,
                if_timing_info
            )

            # Prepare metrics for logging
            original_metrics = {
                'parameters': original_params,
                'accuracy': base_accuracy,
                'inference_time': original_inference_time
            }
            
            default_metrics = {
                'parameters': default_params,
                'accuracy': accuracy_pruned_default * 100,
                'inference_time': default_inference_time,
                'parameter_reduction_percentage': (1 - default_params/original_params) * 100,
                'speed_improvement': (original_inference_time - default_inference_time)/original_inference_time * 100
            }
            
            if_metrics = {
                'parameters': if_params,
                'accuracy': accuracy_pruned_IF * 100,
                'inference_time': if_inference_time,
                'parameter_reduction_percentage': (1 - if_params/original_params) * 100,
                'speed_improvement': (original_inference_time - if_inference_time)/original_inference_time * 100
            }
            
            # Log results
            log_results(log_file, group_idx, sparsity, original_metrics, default_metrics, if_metrics)

            # Print results
            print("\n" + "="*50)
            print(f"Group index: {group_idx}. Sparsity: {sparsity}")
            print("-"*50)
            print("Original Model:")
            print(f"Parameters: {original_params:,}")
            print(f"Accuracy: {base_accuracy:.2f}%")
            print(f"Inference time: {original_inference_time:.2f}ms")
            print("\nDefault Pruning:")
            print(f"Accuracy: {accuracy_pruned_default*100:.2f}%")
            print(f"Parameters: {default_params:,} ({default_params/original_params*100:.1f}% of original)")
            print(f"Parameter reduction: {(original_params-default_params):,} ({(1-default_params/original_params)*100:.1f}%)")
            print(f"Inference time: {default_inference_time:.2f}ms ({default_inference_time/original_inference_time*100:.1f}% of original)")
            print(f"Speed improvement: {(original_inference_time-default_inference_time)/original_inference_time*100:.1f}%")
            print("\nIntra-Fusion:")
            print(f"Accuracy: {accuracy_pruned_IF*100:.2f}%")
            print(f"Parameters: {if_params:,} ({if_params/original_params*100:.1f}% of original)")
            print(f"Parameter reduction: {(original_params-if_params):,} ({(1-if_params/original_params)*100:.1f}%)")
            print(f"Inference time: {if_inference_time:.2f}ms ({if_inference_time/original_inference_time*100:.1f}% of original)")
            print(f"Speed improvement: {(original_inference_time-if_inference_time)/original_inference_time*100:.1f}%")
            print("="*50)

    print(f"\nAll results have been logged to {log_file}")