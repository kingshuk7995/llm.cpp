import struct
import torch
from transformers import GPT2Model

def export_weights(model_name="gpt2", output_path="gpt2_weights.bin"):
    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2Model.from_pretrained(model_name)
    state_dict = model.state_dict()
    
    with open(output_path, "wb") as f:
        def write_tensor(tensor):
            tensor = tensor.float().contiguous()
            f.write(tensor.numpy().tobytes())

        print("Writing embeddings...")
        write_tensor(state_dict["wte.weight"])
        write_tensor(state_dict["wpe.weight"])
        
        num_layers = 12
        hidden_size = 768
        
        for i in range(num_layers):
            print(f"Writing block {i}...")
            prefix = f"h.{i}."
            
            # LayerNorm 1
            write_tensor(state_dict[prefix + "ln_1.weight"])
            write_tensor(state_dict[prefix + "ln_1.bias"])
            
            # Attention
            c_attn_w = state_dict[prefix + "attn.c_attn.weight"] # [768, 2304]
            c_attn_b = state_dict[prefix + "attn.c_attn.bias"]
            
            q_w, k_w, v_w = c_attn_w.split(hidden_size, dim=1)
            q_b, k_b, v_b = c_attn_b.split(hidden_size, dim=0)
            
            write_tensor(q_w)
            write_tensor(q_b)
            write_tensor(k_w)
            write_tensor(k_b)
            write_tensor(v_w)
            write_tensor(v_b)
            
            write_tensor(state_dict[prefix + "attn.c_proj.weight"])
            write_tensor(state_dict[prefix + "attn.c_proj.bias"])
            
            # LayerNorm 2
            write_tensor(state_dict[prefix + "ln_2.weight"])
            write_tensor(state_dict[prefix + "ln_2.bias"])
            
            # MLP
            write_tensor(state_dict[prefix + "mlp.c_fc.weight"])
            write_tensor(state_dict[prefix + "mlp.c_fc.bias"])
            write_tensor(state_dict[prefix + "mlp.c_proj.weight"])
            write_tensor(state_dict[prefix + "mlp.c_proj.bias"])
            
        print("Writing final LayerNorm...")
        write_tensor(state_dict["ln_f.weight"])
        write_tensor(state_dict["ln_f.bias"])
        
        print("Writing LM Head...")
        # In GPT-2, lm_head is tied to wte, but transposed since our model expects [hidden_size, vocab_size]
        write_tensor(state_dict["wte.weight"].T)

    print(f"Done! Exported to {output_path}")

if __name__ == "__main__":
    export_weights()
