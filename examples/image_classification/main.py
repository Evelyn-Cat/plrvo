# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CIFAR-10 classification with Vi-T."""
import logging

import fire
import torch
import torch.nn.functional as F
import tqdm
import transformers
from ml_swissknife import utils
from torchvision import transforms

import os, sys; sys.path.append("..")
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 
os.environ["TRANSFORMERS_CACHE"] = os.path.join(current_dir, "..", "cache")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    xents, zeons = [], []
    for i, (images, labels) in enumerate(loader):
        images, labels = tuple(t.to(device) for t in (images, labels))
        logits = model(pixel_values=images).logits
        xents.append(F.cross_entropy(logits, labels, reduction='none'))
        zeons.append(logits.argmax(dim=-1).ne(labels).float())
    return tuple(torch.cat(lst).mean().item() for lst in (xents, zeons))


def main(
    per_device_train_batch_size=100,
    task_name="cifar10",
    model_name_or_path='vit',
    noise_type="Gaussian",
    config_idx=0,
    output_dir="image_result",
    batch_size=1024,
    test_batch_size=500,
    epochs=10,
    lr=2e-3,
    linear_probe=True,
):  

    import torch
    print(noise_type)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(device)

    config_idx = int(config_idx)
    train_batch_size = batch_size
    gradient_accumulation_steps = train_batch_size // per_device_train_batch_size
    if model_name_or_path=="vit":
        model_name_or_path = "google/vit-base-patch16-224"

     
    import torch
    print(noise_type)
    print(config_idx)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(device)
    
    if noise_type == "PLRVO" or noise_type == "Gaussian":
        print("confirm imported PLRVO")
        from plrvo_transformers import PrivacyEngine 
        from private_transformers import freeze_isolated_params_for_vit
        assert config_idx > 0
    elif noise_type == "Laplace":
        print("confirm imported Laplace")
        from prv_accountant import PRVAccountant # https://github.com/microsoft/prv_accountant
        # https://github.com/google/differential-privacy/blob/main/python/dp_accounting/dp_accounting/pld/privacy_loss_distribution_basic_example.py
    else:
        assert config_idx == 0
        print("confirm running non-private")
        assert noise_type == "non"

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader, test_loader = utils.get_loader(
        data_name=task_name,
        task="classification",
        train_batch_size=per_device_train_batch_size,
        test_batch_size=test_batch_size,
        data_aug=False,
        train_transform=image_transform,
        test_transform=image_transform,
    )

    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = 10
    model = transformers.ViTForImageClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
    ).to(device)
    if linear_probe:
        model.requires_grad_(False)
        model.classifier.requires_grad_(True)
        logging.warning("Linear probe classification head.")
    else:
        freeze_isolated_params_for_vit(model)
        logging.warning("Full fine-tune up to isolated embedding parameters.")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=train_batch_size,
        sample_size=50000,
        epochs=epochs,
        noise_type=noise_type,
        config_idx=config_idx,
    )
    privacy_engine.attach(optimizer)

    global_steps = 0
    train_loss_meter = utils.AvgMeter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for global_step, (images, labels) in pbar:
            global_steps = global_steps + 1

            model.train()
            images, labels = tuple(t.to(device) for t in (images, labels))
            logits = model(pixel_values=images).logits
            loss = F.cross_entropy(logits, labels, reduction="none")
            train_loss_meter.step(loss.mean().item())
            if global_step % gradient_accumulation_steps == 0:
                optimizer.step(loss=loss)
                optimizer.zero_grad()
            else:
                optimizer.virtual_step(loss=loss)
            pbar.set_description(f"Train loss running average: {train_loss_meter.item():.4f}")
        avg_xent, avg_zeon = evaluate(test_loader, model)
        logging.warning(
            f"Epoch: {epoch}, average cross ent loss: {avg_xent:.4f}, average zero one loss: {avg_zeon:.4f}"
        )
        print(f"Running {global_steps} steps.")


if __name__ == "__main__":
    fire.Fire(main)
