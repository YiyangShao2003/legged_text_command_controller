import torch
import json
import numpy as np
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

raw_data = {
    "task5": "a person walks and then makes a right turn",
    "task7": "a man walks forward then stops",
    "task14": "a person wonders in an oval path and ends where he started",
    "task19": "a man runs to the right then runs to the left then back to the middle",
    "task20": "person is walking without their arms swinging",
    "task21": "a person walks in a counterclockwise circle",
    "task23": "a person walks forward then turns around and walks back",
    "task26": "a person paces back and forth in a small space",
    "task28": "a person walking backwards slowly",
    "task32": "a person is pacing the floor back and forth",
    "task36": "person walks up and then spins counter clockwise then walks back",
    "task52": "a person walks to the right in a partial circle",
    "task59": "a person casually walks in a figure 8 pattern",
    "task62": "a person shuffles from the left to the right, then shuffles back to the left",
    "task79": "a person turns left, then turns right, then turns left again",
    "task89": "he is running straight and stopped",
    "task3": "a man lifts something on his left and places it down on his right",
    "task9": "a man waves his right hand",
    "task10": "a man pats his left hand with his right hand",
    "task16": "someone is playing the violin",
    "task29": "man fix his tie in the mirror",
    "task38": "a person bends their arms at the elbow and moves their left arm up and down",
    # "task46": ,
    # "task49": ,
    # "task51": ,
    # "task54": ,
    # "task55": ,
    # "task60": ,
    # "task84": ,
    # "task85": ,
    # "task87": ,
}

dataset = {}

for task_name, caption in raw_data.items():
    cpation_embedding = clip_model.encode_text(clip.tokenize(caption).to(device)).detach().cpu().numpy()
    dataset[task_name] = {
        "caption": caption,
        "embedding": cpation_embedding.tolist()
    }
    
with open("dataset.json", "w") as f:
    json.dump(dataset, f)