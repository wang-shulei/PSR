<h3 align="center">
    <!-- <img src="assets/logo.png" alt="Logo" style="vertical-align: middle; width: 40px; height: 40px;"> -->
    PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards
    <!-- Less-to-More Generalization: </br>Unlocking More Controllability by In-Context Generation -->
</h3>

<p align="center"> 
<a href="https://arxiv.org/abs/2504.02160"><img alt="Build" src="https://img.shields.io/badge/arXiv-1234.51323-b31b1b"></a>
<a href="https://drive.google.com/file/d/1p7I2E-R8q5zN8XVQwU__uKzsPai5jPWZ/view?usp=drive_link"><img src="https://img.shields.io/badge/Dataset-Download-green"></a>
<a href="https://github.com/wang-shulei/PSR"><img alt="Build" src="https://img.shields.io/github/stars/wang-shulei/PSR"></a> 
</p>


<p align="center" style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
    <span>Shulei Wang</span><sup>1,2†</sup>,
    <span>Longhui Wei</span><sup>2‡</sup>,
    <span>Xin He</span><sup>2</sup>,
    <span>Jiangbo Ouyang</span><sup>2</sup>,
    <span>Hui Lu</span><sup>2</sup>,
    <span>Zhou Zhao</span><sup>1</sup>,
    <span>Qi Tian</span><sup>2*</sup>
    <br>
    <span style="font-size: 16px;">
        <sup>1</sup> Zhejiang University &nbsp;&nbsp;
        <sup>2</sup> Huawei Inc.
    </span>
    <br>
    <span style="font-size: 14px; font-style: italic; color: #555;">
        († Work was done during internship at Huawei, ‡ Project Leader, * Corresponding author.)
    </span>
</p>
<p align="center">
<img src="./assets/quan_radar.jpeg" width=95% height=95% 
class="center">
</p>

## 📖 Introduction
Multi-subject personalized image generation remains challenging, as existing models often fail to maintain subject identity, handle complex compositions, or follow detailed text instructions. To address these issues, we introduce a scalable multi-subject data generation pipeline and a reinforcement learning framework powered by Pairwise Subject-Consistency Reward (PSR).

We further construct PSRBench, a fine-grained benchmark covering seven subsets to evaluate subject consistency, semantic alignment, and visual quality.
Our approach achieves strong improvements across all dimensions, delivering more reliable and controllable multi-subject personalized generation.
<p align="center">
<img src="./assets/methods.jpeg" width=95% height=95% 
class="center">
</p>

## 🧪 PSRBench: Fine-grained Evaluation for Multi-Subject Personalization
Overview of PSRBench, PSRBench consists of seven subsets, each evaluating multi-subject personalized generation models from a different perspective. Each subset contains 50 high-quality test samples and is evaluated along three distinct dimensions.
<p align="center">
<img src="./assets/PSRBench.jpeg" width=95% height=95% 
class="center">
</p>


## 📊 PSRBench Evaluation

To evaluate your model on **PSRBench**, follow the steps below:


### **1. Download PSRBench**

You can download the full benchmark (including prompts, reference images) from the link below:

👉 **Download PSRBench**
 *https://drive.google.com/file/d/1p7I2E-R8q5zN8XVQwU__uKzsPai5jPWZ/view?usp=drive_link*

After downloading, extract the dataset and ensure the folder structure follows:

```
PSRBench/
  ├── Attribute.json
  ├── Action.json
  ├── Background.json
  ├── Position.json
  ├── Complex.json
  ├── Three.json
  ├── Four.json
  ├── subjects/
  └── animals/
  
```

------

### **2. Generate Images for Each Sample**

Then run your image generation model and save the generated image:

```
result = your_model(
    prompt=item["instruction"],
    ref_images=item["input_paths"]
)

output_path = f"outputs/{subset}/{sample_id}.png"
result.save(output_path)

# Save the output path back to the original JSON entry
item["result_path"] = output_path
```
The generated image will be stored in:

```
outputs/{subset}/{sample_id}.png
```

And the updated JSON entry (with `result_path`) can later be written to your results file for evaluation.



------


### **3. Run Evaluation**
After generation is complete, create a new environment for evaluation:
```
conda create -n PSR python=3.10
conda activate PSR
pip install -r requirements.txt
```
Install Grounding DINO:
```
pip install --no-build-isolation -e grounding_dino
```

To evaluate **Subject Consistency**, run:

```
python ./eval/subject-consistency/eval-2-subject-sc.py \
    --json_path path_to_your_result_json \
    --data_root path_data_root
```

To evaluate **Semantic Alignment**, for position subset, run:

```
python ./eval/semantic-alignmnet/postion-alignment.py \
    --json_path path_to_your_result_json \
    --data_root path_data_root
```
For other subject, you shuold first create a new environment and install sglang:

```
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"
```
Then launch a server
```
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-32B-Instruct --api-key psr --port 17140 --chat-template qwen2-vl --enable-p2p-check --mem-fraction-static 0.85
```
Then you can run evaluation scripts
```
python ./eval/semantic-alignmnet/postion-alignment.py \
    --json_path path_to_your_result_json \
    --data_root path_data_root
```

To evaluate **Aesthetic Preference**, follow the HPSv3 scripts from
https://github.com/MizzenAI/HPSv3


## 🖼️ Visualizations
Visualization results on PSRBench
<p align="center">
<img src="./assets/qualitative.jpeg" width=95% height=95% 
class="center">
</p>
Visualization results on DreamBench
<p align="center">
<img src="./assets/dreambench_result.jpeg" width=95% height=95% 
class="center">
</p>

## 📘 Citation 
If you find PSR helpful in your research or applications, please consider starring the project ⭐ and citing it using the following BibTeX entry.

```
TBD
```

<!-- ## ⚡️ Quick Start

### 🔧 Requirements and Installation

Install the requirements
```bash
# pip install -r requirements.txt # legacy installation command

## create a virtual environment with python >= 3.10 <= 3.12, like
# python -m venv uno_env
# source uno_env/bin/activate
# or
# conda create -n uno_env python=3.10 -y
# conda activate uno_env
# then install the requirements by you need

# !!! if you are using amd GPU/NV RTX50 series/macos MPS, you should install the correct torch version by yourself first
# !!! then run the install command
pip install -e .  # for who wanna to run the demo/inference only
pip install -e .[train]  # for who also want to train the model
```

then download checkpoints in one of the three ways:
1. Directly run the inference scripts, the checkpoints will be downloaded automatically by the `hf_hub_download` function in the code to your `$HF_HOME`(the default value is `~/.cache/huggingface`).
2. use `huggingface-cli download <repo name>` to download `black-forest-labs/FLUX.1-dev`, `xlabs-ai/xflux_text_encoders`, `openai/clip-vit-large-patch14`, `bytedance-research/UNO`, then run the inference scripts. You can just download the checkpoint in need only to speed up your set up and save your disk space. i.e. for `black-forest-labs/FLUX.1-dev` use `huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors` and `huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors`, ignoreing the text encoder in `black-forest-labes/FLUX.1-dev` model repo(They are here for `diffusers` call). All of the checkpoints will take 37 GB of disk space.
3. use `huggingface-cli download <repo name> --local-dir <LOCAL_DIR>` to download all the checkpoints mentioned in 2. to the directories your want. Then set the environment variable `AE`, `FLUX_DEV`(or `FLUX_DEV_FP8` if you use fp8 mode), `T5`, `CLIP`, `LORA` to the corresponding paths. Finally, run the inference scripts.
4. **If you already have some of the checkpoints**, you can set the environment variable `AE`, `FLUX_DEV`, `T5`, `CLIP`, `LORA` to the corresponding paths. Finally, run the inference scripts. -->
