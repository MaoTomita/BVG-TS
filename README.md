# BVG-LS: Bias-Variance Guided Layer Selection

## Links
- [Paper: "A Simple Finetuning Strategy Based on Bias-Variance Ratios of Layer-Wise Gradients"](https://openaccess.thecvf.com/content/ACCV2024/papers/Tomita_A_Simple_Finetuning_Strategy_Based_on_Bias-Variance_Ratios_of_Layer-Wise_ACCV_2024_paper.pdf)
- [Project Page](http://www.vip.sc.e.titech.ac.jp/proj/BVGLS/)

## Repository Structure

- **`BVG_LS.py`**: Core implementation of the BVG_LS class.
- **`train.py`**: Sample training script demonstrating the use of BVG_LS with models and datasets.

## File Descriptions

### BVG_LS.py

This file implements the BVG_LS class. Key functions:

- **`set_lr(global_lr, n_update)`**: Adjusts learning rates of layers based on their bias-variance ratios.
- **`_cal_bvr()`**: Calculates the bias-variance ratio for each layer (used internally within `set_lr` and not intended for standalone use).

#### How to Use BVG_LS

1. **Prepare an Optimizer with Layer-wise Parameter Groups:**
   Split your model's parameters into groups corresponding to different layers or modules.

   ```python
   # Example for WideResNet-50-2
   layers = [
       nn.Sequential(model.conv1, model.bn1),
       model.layer1, model.layer2, model.layer3, model.layer4, model.fc
   ]
   param_groups = [{'params': layer.parameters()} for layer in layers]
   optimizer = torch.optim.SGD(param_groups, lr=0.01, momentum=0.9)
   ```

2. **Initialize a BVG_LS Instance:**
   Pass the optimizer to the BVG_LS constructor.

   ```python
   bvg_ls = BVG_LS(optimizer)
   ```

3. **Update Learning Rates During Training:**
   Inside your training loop, use BVG_LS to adjust the learning rates of layers dynamically.

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

       # Adjust learning rates using BVG_LS
       bvg_ls.set_lr(global_lr=0.01, n_update=1)

       # Update parameters
       optimizer.step()
   ```

### train.py

A sample script that demonstrates:

- Dataset loading and preprocessing.
- Model preparation for WideResNet-50-2 and Vision Transformers.
- Finetuning using BVG_LS.

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


