# BVG-TS: Bias-Variance Guided Tensor Selection

## Repository Structure

- **`BVG_TS.py`**: Core implementation of the BVG_TS class.
- **`train.py`**: Sample training script demonstrating the use of BVG_TS with models and datasets.

## File Descriptions

### BVG_TS.py

This file implements the BVG_TS class. Key functions:

- **`set_lr(global_lr, update_ratio)`**: Adjusts learning rates of tensors based on their bias-variance ratios.
- **`_cal_bvr()`**: Calculates the bias-variance ratio for each tensor (used internally within `set_lr` and not intended for standalone use).

#### How to Use BVG_TS

1. **Prepare an Optimizer with Tensor-wise Parameter Groups:**
   Split your model's parameters into groups corresponding to different tensors.

   ```python
   # Example for WideResNet-50-2
   model = models.wide_resnet50_2(weights='DEFAULT')
   model.fc = nn.Linear(model.fc.in_features, num_classes)
   
   param_groups = [{'params': [param]} for name, param in model.named_parameters()]
   
   optimizer = torch.optim.SGD(param_groups, lr=0.001, momentum=0.9)
   ```

2. **Initialize a BVG_TS Instance:**
   Pass the optimizer to the BVG_TS constructor.

   ```python
   bvg_ts = BVG_TS(optimizer)
   ```

3. **Update Learning Rates During Training:**
   Inside your training loop, use BVG_TS to adjust the learning rates of tensors dynamically.

   ```python
   for inputs, targets in train_loader:
       inputs, targets = inputs.to(device), targets.to(device)

       # Zero gradients
       optimizer.zero_grad()

       # Forward pass
       outputs = model(inputs)
       loss = criterion(outputs, targets)

       # Backward pass
       loss.backward()

       # Adjust learning rates using BVG_TS
       bvg_ts.set_lr(global_lr=0.001, update_rate=0.5)

       # Update parameters
       optimizer.step()
   ```

### train.py

A sample script that demonstrates:

- Dataset loading and preprocessing.
- Model preparation for WideResNet-50-2 and Vision Transformers.
- Finetuning using BVG_TS.

Supported dataset (`--dataset`):

- Oxford-IIIT Pet Dataset (`pets`)
- Describable Textures Dataset (`dtd`)
- CIFAR-100 (`cifar100`) 

Supported network (`--network`):

- WideResNet-50-2 (`wrn_50_2`)
- Vision Transformer Small (`vit_small`)

#### Running the Script

Use the following command to run the training script:

```bash
python train.py --dataset <dataset_name> --network <network_name> --batch_size <batch_size> --epochs <epochs> --lr <learning_rate>
```

Example:

```bash
python train.py --dataset pets --network wrn_50_2 --batch_size 128 --epochs 20 --lr 0.01
```

To enable logging with Weights & Biases, include the `--use_wandb` flag:

```bash
python train.py --dataset pets --network vit_small --use_wandb
```


