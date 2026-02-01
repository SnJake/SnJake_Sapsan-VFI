![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Made for ComfyUI](https://img.shields.io/badge/Made%20for-ComfyUI-blueviolet)

SnJake **Sapsan-VFI** is a custom ComfyUI node for **x2 video frame interpolation**. It inserts one in-between frame for every input pair, doubling the FPS.

---

# Examples


https://github.com/user-attachments/assets/2caa951b-a6c9-4423-8a5e-364b65b88e6b




https://github.com/user-attachments/assets/55b9cc81-acd6-4bde-bd5e-66b082c34c10




https://github.com/user-attachments/assets/fd9ebc75-20fa-434f-9541-1075558b22ac


---

# Installation

The installation consists of two steps: installing the node and making the weights available.

## Step 1: Install the Node

1. Open a terminal or command prompt.
2. Navigate to your ComfyUI `custom_nodes` directory.
   ```bash
   # Example for Windows
   cd D:\ComfyUI\custom_nodes\

   # Example for Linux
   cd ~/ComfyUI/custom_nodes/
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/SnJake/SnJake_Sapsan-VFI.git
   ```
4. For standard ComfyUI installations (with venv):
    1. Make sure your ComfyUI virtual environment (`venv`) is activated.
    2. Navigate into the new node directory and install the requirements (if needed):
       ```bash
       cd SnJake_Sapsan-VFI
       pip install -r requirements.txt
       ```
   For Portable ComfyUI installations:
    1. Navigate back to the **root** of your portable ComfyUI directory (e.g., `D:\ComfyUI_windows_portable`).
    2. Run the following command to use the embedded Python to install the requirements. *Do not activate any venv.*
       ```bash
       python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\SnJake_Sapsan-VFI\requirements.txt
       ```

## Step 2: Model Weights

On first use the node can automatically download weights from the repository.

- Default weights location: `ComfyUI/models/sapsan_vfi/`
- Available weights on HF: `Sapsan-VFI.safetensors` and `Sapsan-VFI.pt`

If you want to download manually:
1. Download the weights from the HF repo: `https://huggingface.co/SnJake/Sapsan-VFI`.
2. Place the file(s) into `ComfyUI/models/sapsan_vfi/`.

## Step 3: Restart

Restart ComfyUI completely. The node will appear under **`😎 SnJake/VFI`**.

---

# Usage

The node menu path is **`😎 SnJake/VFI`**.

## Inputs

- `weights_name`: Select weights from the dropdown (auto-download if missing).
- `images`: Input frame batch (`IMAGE`) from your video loader.
- `fps` (optional): Input FPS from `GetVideoComponents` (used to output x2 FPS).
- `t`: Interpolation time (default `0.5`).
- `pad_multiple`: Padding multiple (0 = use `config.yaml`).
- `match_brightness`: Match predicted frame brightness to inputs (reduces flicker).
- `amp`: Precision (`auto`, `bf16`, `fp16`, `none`).
- `device`: Device (`auto`, `cuda`, `cpu`).
- `torch_compile`: Enable `torch.compile` (optional optimization).
- `console_progress`: Print progress to the ComfyUI console.

## Outputs

- `images`: Interpolated frames (x2 length minus 1).
- `fps`: Output FPS (input FPS * 2, or 0 if not provided).

---

# Training Details

- Created out of curiosity and personal interest.
- Total epochs: **11**
- Dataset: **2700 videos**
- Shards: **151** shards of **1000** shadrs in each. 151 000 triplets.

Training code is included in `training_code/` for reference.

---

# Disclaimer

This project was made purely for curiosity and personal interest. The code was written by GPT-5.2 Codex.

---

# License

MIT.
