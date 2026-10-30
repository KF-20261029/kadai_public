# Copyright 2026 SO-101 task contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restore the public pi0.5 base and this project's trained pen adapter."""

import argparse
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from peft import PeftModel
import torch
from vlash.policies.pi05.configuration_pi05 import PI05Config
from vlash.policies.pi05.modeling_pi05 import PI05Policy


def load_policy(base_model, adapter, use_lora=True):
    """Load inference weights and restore the saved state/action scaling."""
    base_model = Path(base_model)
    adapter = Path(adapter)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this policy configuration.')

    config = PreTrainedConfig.from_pretrained(str(base_model), local_files_only=True)
    if not isinstance(config, PI05Config):
        raise ValueError('The base model must use the pi0.5 configuration.')
    config.device = 'cuda'
    config.compile_model = False
    config.fuse_qkv = False
    config.fuse_gate_up = False
    config.empty_cameras = 1
    config.n_action_steps = 50

    print('Loading the base model...', flush=True)
    policy = PI05Policy.from_pretrained(
        str(base_model), config=config, local_files_only=True,
    )
    if use_lora:
        print('Restoring the trained LoRA weights...', flush=True)
        adapted = PeftModel.from_pretrained(
            policy.model, str(adapter / 'lora_adapters'),
            autocast_adapter_dtype=False, local_files_only=True,
        )
        policy.model = adapted.merge_and_unload(safe_merge=True)
    else:
        print('Using the unmodified base weights.', flush=True)

    statistics = torch.load(
        adapter / 'normalize_buffers.pt', map_location='cpu', weights_only=True,
    )
    destinations = {
        'observation.state': ('normalize_inputs',),
        'action': ('normalize_targets', 'unnormalize_outputs'),
    }
    if set(statistics) != set(destinations):
        raise ValueError('Saved scaling must include state and action statistics.')
    with torch.no_grad():
        for feature, modules in destinations.items():
            entry = statistics[feature]
            if not all(torch.isfinite(entry[key]).all() for key in ('mean', 'std')):
                raise ValueError('Saved scaling contains non-finite values.')
            if not (entry['std'] > 0).all():
                raise ValueError('Saved standard deviations must be positive.')
            attribute = 'buffer_' + feature.replace('.', '_')
            for module in modules:
                target = getattr(getattr(policy, module), attribute)
                for key in ('mean', 'std'):
                    buffer = getattr(target, key)
                    if entry[key].shape != buffer.shape:
                        raise ValueError('Saved scaling has an incompatible shape.')
                    buffer.copy_(entry[key].to(buffer))
    return policy.eval().requires_grad_(False)


def main():
    """Check that the task's inference model can be restored on the GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, default=Path('/cache/model'))
    parser.add_argument('--adapter', type=Path, default=Path('/workspace/models/pen_lora'))
    args = parser.parse_args()
    policy = load_policy(args.base_model, args.adapter)
    torch.cuda.synchronize()
    parameter = next(policy.model.parameters())
    print('Model ready.', flush=True)
    print('Device:', parameter.device)
    dtypes = sorted({str(value.dtype) for value in policy.model.parameters()})
    print('Parameter dtypes:', ', '.join(dtypes))
    print('State and action scaling restored.', flush=True)


if __name__ == '__main__':
    main()
