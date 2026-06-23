import torch
import argparse
import os

def split_mrcps_weights(ckpt_path, save_path1, save_path2):
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle both full state dict and raw state dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    branch1_dict = {}
    branch2_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith('branch1.'):
            # ModelMRCPS_Branch1 expects the prefix 'branch.'
            new_key = k.replace('branch1.', 'branch.', 1)
            branch1_dict[new_key] = v
        elif k.startswith('branch2.'):
            # ModelMRCPS_Branch2 expects the prefix 'branch.'
            new_key = k.replace('branch2.', 'branch.', 1)
            branch2_dict[new_key] = v
            
    # Create directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(save_path1)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(save_path2)), exist_ok=True)
            
    # Save weights
    torch.save(branch1_dict, save_path1)
    print(f"Saved Branch 1 weights to {save_path1} (Total keys: {len(branch1_dict)})")
    
    torch.save(branch2_dict, save_path2)
    print(f"Saved Branch 2 weights to {save_path2} (Total keys: {len(branch2_dict)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dual-branch ModelMRCPS weights into two single-branch weights.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to original dual-branch model checkpoint (e.g., weights/model_best.pth)")
    parser.add_argument("--out1", "-o1", type=str, required=True, help="Path to save branch 1 weights")
    parser.add_argument("--out2", "-o2", type=str, required=True, help="Path to save branch 2 weights")
    
    args = parser.parse_args()
    
    split_mrcps_weights(args.input, args.out1, args.out2)
