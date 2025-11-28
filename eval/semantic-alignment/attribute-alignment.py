from PIL import Image
import torch
import re
import base64
from io import BytesIO
import json
from openai import OpenAI
import base64
client = OpenAI(base_url="http://localhost:17140/v1", api_key="psr")
# def pil_image_to_base64(image):
#     buffered = BytesIO()
#     image.save(buffered,format="PNG")
#     encoded_image_text = base64.b64decode(buffered.getvalue()).decode("utf-8")
#     base64_qwen = f"data:image;base64,{encoded_image_text}"
#     return base64_qwen
def image_to_data_url(path):
    with open(path,"rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"
def extract_score(text):
    pattern = r"\[Score\]:\s*([\d.]+)"
    matches=re.search(pattern,text)
    if matches:
        return float(matches.group(1))
    else:
        return 5
class QwenVLScorer(torch.nn.Module):
    def __init__(self,device="cuda",dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.task='''
        Your role is to evaluate whether the attributes of objects and attribute description in the image are consistent.
        Rule1: The score range is 0-10, where 10 indicates that all attributes of each object are completely consistent with the attribute description.
        A lower score indicates low consistency.
        Rule2: You only need to focus on the degree of matching between the attributes of subjects of image and the attribute description.
        Rule3: The criteria for judging attributes should be relatively stringent.
        Rule4: Please first provide a detailed analysis of the evaluation process, including the criteria for judging attribute alignment, then give a final
        Score from 0 to 10. The output text must follow the format [Thought]: … [Score]: …
        [attribute description]: {}
        [Thought]:
        [Score]:
        '''
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        image_urls = [image_to_data_url(image) for image in images]
        scores = []
        for image_url,p in zip(image_urls,prompt):
            prompt_ac=self.task.format(p)
            response=client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {
                        "role":"user",
                        "content":[
                            {"type":"image_url","image_url":{"url":image_url}},
                            {"type":"text","text":prompt_ac},
                        ],
                    }
                ],
                temperature=0.2,
            )
            score=extract_score(response.choices[0].message.content)
            scores.append(score)
        return scores
def main():
    scorer=QwenVLScorer(
        device="cuda",
        dtype=torch.bfloat16
    )
    import json
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the input JSON file, e.g., attr.json")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for data loading")
    args = parser.parse_args()
    with open(args.json_path,"r",encoding="utf-8") as f:
        data=json.load(f)
    results=[]
    for idx,item in enumerate(data):
        try:
            img_path=item["result_path"]
            ins=item["instruction"]
            prompt=[ins]
            pil_images=[img_path]
            score=scorer(prompt,pil_images)
            results.extend(score)
        except:
            print("{} don't exist".format(idx))
    print("final score: ",sum(results)/len(results))
if __name__=="__main__":
    main()