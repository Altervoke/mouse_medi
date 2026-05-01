import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.generation.generator import load_foundation_model, generate_pixel_medi_func

def main():
    device = 'cuda'
    
    examples = [
        ('V1', 'L4', 9, 3, 6900),
        ('LM', 'L5', 5, 6, 6184),
        ('RL', 'L4', 5, 7, 3838),
        ('AL', 'L23', 6, 7, 4143),
    ]
    
    loaded_models = {}
    
    for area, layer, session, scan, readout_id in examples:
        key = (session, scan)
        if key not in loaded_models:
            loaded_models[key] = load_foundation_model(session, scan, device=device)
            
        model = loaded_models[key]
        
        generate_pixel_medi_func(
            session=session,
            scan=scan,
            readout_id=readout_id,
            brain_area=area,
            layer=layer,
            model=model,
            device=device
        )

if __name__ == "__main__":
    main()
