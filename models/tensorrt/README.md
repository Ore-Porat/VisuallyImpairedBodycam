# Fast grocery TensorRT engines

`config/server.yaml` expects two TensorRT engines in this folder:

| Engine | Built from | Purpose | Export size |
|---|---|---|---|
| `shelf_world_s.engine` | `models/yolov8s-worldv2.pt` | YOLO-World S product detector, FP16, 640 px | ~102 MB |
| `hand_egohands_960.engine` | `models/hand_best.pt` | Hand detector, FP16, 960 px | ~125 MB |

They are **not committed** because TensorRT engines are specific to the GPU, TensorRT
version, driver and export flags that built them. A fresh clone must build or copy them.
`models/**/*.engine` is gitignored.

## Build the engines from the `.pt` models

Run these commands from the repository root on the laptop that will run the server.

### 1. Check the environment

TensorRT export needs a CUDA-capable GPU and the `ultralytics` package:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available())'
```

The last command must print `True`.

On NixOS, enter the project shell first (it exports the NVIDIA driver libraries):

```bash
nix-shell
source .venv/bin/activate
```

If you use the repo virtual environment directly, replace `python` below with
`.venv/bin/python`.

Install the export-only dependencies if they are missing:

```bash
pip install onnx onnxslim 'nvidia-modelopt[onnx]>=0.44'
```

`tensorrt-cu13` is already part of the server extra (`pip install -e ".[server]"`). The
ModelOpt package is only needed for FP16 export with TensorRT 11.

### 2. Build the product detector

```bash
python models/export_tensorrt.py \
  --skip-hand \
  --world-model models/yolov8s-worldv2.pt \
  --world-out shelf_world_s.engine \
  --imgsz-world 640
```

This bakes the default generic product prompts into the YOLO-World model, then writes
`models/tensorrt/shelf_world_s.engine`.

### 3. Build the hand detector

```bash
python models/export_tensorrt.py \
  --skip-world \
  --hand-model models/hand_best.pt \
  --hand-out hand_egohands_960.engine \
  --imgsz-hand 960
```

This writes `models/tensorrt/hand_egohands_960.engine`.

### 4. Build both at once

The two steps above can be combined:

```bash
python models/export_tensorrt.py \
  --world-model models/yolov8s-worldv2.pt \
  --world-out shelf_world_s.engine \
  --imgsz-world 640 \
  --hand-model models/hand_best.pt \
  --hand-out hand_egohands_960.engine \
  --imgsz-hand 960
```

### 5. Check the result

```bash
ls -lh models/tensorrt/*.engine
```

The default output folder is `models/tensorrt/`, so no config change is needed.
`config/server.yaml` already points at:

```yaml
model:
  fast_detector_engine: models/tensorrt/shelf_world_s.engine
  fast_hand_engine: models/tensorrt/hand_egohands_960.engine
```

Start the server and check the model line in the log:

```bash
python slr.py server
```

## Optional higher-recall detector

If you also have `models/yolov8l-worldv2.pt`, build the larger detector for better recall at
the cost of some FPS:

```bash
python models/export_tensorrt.py \
  --skip-hand \
  --world-model models/yolov8l-worldv2.pt \
  --world-out shelf_world.engine \
  --imgsz-world 640
```

Then change `model.fast_detector_engine` in `config/server.yaml` to:

```yaml
  fast_detector_engine: models/tensorrt/shelf_world.engine
```

## Export options and troubleshooting

- `--device 0` selects GPU 0; it is the default.
- `--workspace 4.0` sets the TensorRT workspace in GiB; it is the default.
- `--no-half` builds an FP32 engine. Use it if FP16 export fails, but the file will be larger
  and inference slower.
- Override the baked detector prompts with, for example,
  `--detector-classes "product,package,can,bottle,box,bag,jar,container"`.
- Rebuild both engines after changing the GPU, TensorRT version, `.pt` weights, image size,
  precision, or detector classes. An engine built for another machine or TensorRT version may
  fail to deserialize.
- Copying engines from another checkout only works when that checkout used the same GPU and
  TensorRT version. Otherwise rebuild from the `.pt` files above.

## CLIP

CLIP (`openai/clip-vit-base-patch16`) is not stored here. It is downloaded from Hugging Face
once and cached under `~/.cache/huggingface`. If the laptop must run offline, pre-download it:

```bash
python -c 'from transformers import CLIPModel, CLIPProcessor; \
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"); \
CLIPModel.from_pretrained("openai/clip-vit-base-patch16")'
```

## CUDA check

TensorRT engines need CUDA. From the VIB Nix shell:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

If it prints `False`, make sure `shell.nix` exports
`LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH`.
