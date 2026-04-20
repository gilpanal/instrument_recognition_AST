# GPU Backend Implementation Plan

This note records the agreed implementation plan for improving device support in this
repository, with particular focus on Apple Silicon (`mps`) and NVIDIA CUDA execution.

The goal is to make the AST inference path work cleanly across:

- `cuda` on NVIDIA GPUs
- `mps` on Apple Silicon GPUs
- `cpu` as the default fallback

This file is intentionally practical and code-oriented so it can be used later during
implementation or by other researchers working on the repository.

---

## 1. Current situation

The current code assumes CUDA in a few places:

- `AST/src/models/ast_models.py` decorates `forward()` with `@autocast('cuda')`
- `AST/instrument_recognition.py` selects only between `cuda` and `cpu`
- `AST/instrument_recognition.py` moves waveform tensors to `device` before calling
  `torchaudio.compliance.kaldi.fbank`
- `AST/instrument_recognition.py` always loads the model through `torch.nn.DataParallel`

This causes two practical issues:

1. On non-CUDA systems, a CUDA autocast warning is emitted:

   > `UserWarning: CUDA is not available or torch_xla is imported. Disabling autocast.`

2. The code is not prepared for Apple GPU execution through PyTorch's `mps` backend.

---

## 2. Important factual constraint

For Apple GPUs, PyTorch uses `mps`, not `cuda`.

In one verified environment:

- `torch.cuda.is_available() == False`
- `torch.backends.mps.is_built() == True`
- `torch.backends.mps.is_available() == False`

MPS availability is environment-specific: it depends on the installed PyTorch build,
the macOS version, and the runtime conditions of the machine where the code runs.
The value of `torch.backends.mps.is_available()` may differ on other Apple Silicon
machines or in other runtime environments.

The planned changes are immediately relevant for cross-platform support, but actual
Apple GPU execution will only occur when `torch.backends.mps.is_available()` returns
`True` on the target machine/runtime.

---

## 3. Agreed implementation changes

### 3.1 Remove the CUDA-only autocast decorator

**File:** `AST/src/models/ast_models.py`

Current issue:

```python
@autocast('cuda')
def forward(self, x):
```

This hardcodes CUDA AMP at the model definition level and emits warnings on systems
without CUDA.

Planned change:

- remove the `@autocast('cuda')` decorator entirely

Rationale:

- CUDA autocast is already handled at the inference call site in
  `AST/instrument_recognition.py`
- model code should not force a backend-specific AMP policy
- removing the decorator makes the model safer on CPU and MPS

---

### 3.2 Extend device selection to support `mps`

**File:** `AST/instrument_recognition.py`

Current issue:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Planned change:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Rationale:

- CUDA remains preferred on NVIDIA systems
- MPS becomes available on Apple Silicon when supported by the runtime
- CPU remains the safe fallback

The `hasattr(...)` guard keeps the code robust across PyTorch builds.

---

### 3.3 Keep `torchaudio.compliance.kaldi.fbank` on CPU

**File:** `AST/instrument_recognition.py`

Current issue in `make_features()`:

```python
audio_tensor = torch.tensor(audio_sr16)
audio_tensor = audio_tensor.to(device)
audio_tensor = audio_tensor.unsqueeze(0)

fbank = torchaudio.compliance.kaldi.fbank(...)
```

Problem:

- `torchaudio.compliance.kaldi.fbank` should be treated as CPU-side preprocessing here
- moving the waveform tensor to `mps` before `fbank` is not a safe assumption
- this is a likely failure point for Apple GPU support

Planned change:

- keep waveform tensor creation on CPU
- compute `fbank` on CPU
- only move the final feature tensor to the selected device later in the inference path

In practice:

```python
audio_tensor = torch.tensor(audio_sr16).unsqueeze(0)
fbank = torchaudio.compliance.kaldi.fbank(...)
```

Then later:

```python
feats_data = feats.expand(1, input_tdim, 128).to(device)
```

Rationale:

- preprocessing remains backend-safe
- model inference still benefits from GPU when available

---

### 3.4 Avoid `DataParallel` outside CUDA

**File:** `AST/instrument_recognition.py`

Current issue in `load_ast_model()`:

```python
audiomodel = torch.nn.DataParallel(ast_mdl, device_ids=[0])
audiomodel.load_state_dict(checkpoint)
```

Problem:

- `DataParallel` is CUDA-oriented here
- it should not be used for MPS or CPU
- the checkpoint was saved from a `DataParallel`-wrapped model, so non-CUDA loading
  must strip the `module.` prefix from parameter names

Planned change:

```python
checkpoint = torch.load(checkpoint_path, map_location=device)

if device.type == "cuda":
    audiomodel = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audiomodel.load_state_dict(checkpoint)
else:
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in checkpoint.items()}
    ast_mdl.load_state_dict(state_dict)
    audiomodel = ast_mdl
```

Rationale:

- preserves current CUDA behavior
- allows clean loading on CPU and MPS

---

### 3.5 Keep AMP limited to CUDA unless MPS is explicitly validated

**File:** `AST/instrument_recognition.py`

Current behavior:

```python
with torch.no_grad():
    if device.type == "cuda":
        with autocast("cuda"):
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
    else:
        output = torch.sigmoid(audio_model.forward(feats_data))
```

Assessment:

- this is a reasonable policy
- CUDA AMP can stay
- MPS should first be made correct and stable before considering AMP there

Planned action:

- do not introduce MPS autocast by default in the first pass

Rationale:

- correctness and portability first
- mixed precision on MPS can be evaluated separately later if needed

---

### 3.6 Update the README device note

**File:** `README.md`

The README should reflect the intended backend behavior:

- CUDA is used automatically on supported NVIDIA systems
- MPS can be used on supported Apple Silicon systems
- CPU fallback remains supported

It should also avoid implying that Apple GPU support is always available merely because
the machine has Apple Silicon. Availability depends on the installed PyTorch build and
runtime conditions.

---

## 4. Suggested implementation order

1. Remove the CUDA-only decorator in `AST/src/models/ast_models.py`
2. Add `mps` device selection in `AST/instrument_recognition.py`
3. Keep `fbank` preprocessing on CPU
4. Make checkpoint loading backend-aware (`cuda` vs non-`cuda`)
5. Verify inference on CPU
6. Verify inference on CUDA if available
7. Verify inference on MPS when `torch.backends.mps.is_available()` is `True`
8. Update README wording

---

## 5. Validation checklist

After implementation, verify:

- no CUDA autocast warning appears on CPU-only or MPS systems
- CPU inference still works and produces the same outputs as before
- CUDA inference still works
- MPS inference works on a machine where `torch.backends.mps.is_available()` is `True`
- the checkpoint loads correctly on all supported backends
- README backend notes match the actual behavior

---

## 6. Expected outcome

After these changes:

- the codebase will stop assuming CUDA-only execution
- Apple Silicon systems will be able to use MPS when the backend is available
- CPU fallback will remain stable
- backend behavior will be clearer both in code and in documentation

This is a portability and usability improvement. It does not guarantee a large speedup
on every Apple machine, because M-series CPUs are already strong and the workload here is
single-stem inference with preprocessing overhead. The main benefit is correct and clean
backend support across platforms.
