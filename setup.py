# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed library

To build wheels on Windows:
1. Install pytorch, such as pytorch 2.3 + cuda 12.1.
2. Install visual cpp build tool.
3. Include cuda toolkit.
4. Launch cmd console with Administrator privilege for creating required symlink folders.


Create a new wheel via the following command:
build_win.bat

The wheel will be located at: dist/*.whl
"""

import os
import torch
from deepspeed import __version__ as ds_version
from setuptools import setup, find_packages
from setuptools.command import egg_info
import time
import typing
import shlex

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

from op_builder import get_default_compute_capabilities, OpBuilder
from op_builder.all_ops import ALL_OPS, accelerator_name
from op_builder.builder import installed_cuda_version

from accelerator import get_accelerator

# Fetch rocm state.
is_rocm_pytorch = OpBuilder.is_rocm_pytorch()
rocm_version = OpBuilder.installed_rocm_version()

RED_START = '\033[31m'
RED_END = '\033[0m'
ERROR = f"{RED_START} [ERROR] {RED_END}"


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def is_env_set(key):
    """
    Checks if an environment variable is set and not "".
    """
    return bool(os.environ.get(key, None))


def get_env_if_set(key, default: typing.Any = ""):
    """
    Returns an environment variable if it is set and not "",
    otherwise returns a default value. In contrast, the fallback
    parameter of os.environ.get() is skipped if the variable is set to "".
    """
    return os.environ.get(key, None) or default


install_requires = fetch_requirements('requirements/requirements.txt')
extras_require = {
    '1bit': [],  # add cupy based on cuda/rocm version
    '1bit_mpi': fetch_requirements('requirements/requirements-1bit-mpi.txt'),
    'readthedocs': fetch_requirements('requirements/requirements-readthedocs.txt'),
    'dev': fetch_requirements('requirements/requirements-dev.txt'),
    'autotuning': fetch_requirements('requirements/requirements-autotuning.txt'),
    'autotuning_ml': fetch_requirements('requirements/requirements-autotuning-ml.txt'),
    'sparse_attn': fetch_requirements('requirements/requirements-sparse_attn.txt'),
    'sparse': fetch_requirements('requirements/requirements-sparse_pruning.txt'),
    'inf': fetch_requirements('requirements/requirements-inf.txt'),
    'sd': fetch_requirements('requirements/requirements-sd.txt'),
    'triton': fetch_requirements('requirements/requirements-triton.txt'),
}

onebit_adam_requires = fetch_requirements('requirements/requirements-1bit-adam.txt')
if torch.cuda.is_available():
    onebit_adam_requires.append(f"cupy-cuda{torch.version.cuda.replace('.','')[:3]}")
install_requires += onebit_adam_requires

# Build environment variables for custom builds
DS_BUILD_LAMB_MASK = 1
DS_BUILD_TRANSFORMER_MASK = 10
DS_BUILD_SPARSE_ATTN_MASK = 100

# Add specific cupy version to both onebit extension variants.
if torch_available and get_accelerator().device_name() == 'cuda':
    cupy = None
    if is_rocm_pytorch:
        rocm_major, rocm_minor = rocm_version
        # cupy support for rocm>5.0 is not available yet.
        if (rocm_major == 5 and rocm_minor == 0) or rocm_major <= 4:
            cupy = f"cupy-rocm-{rocm_major}-{rocm_minor}"
    else:
        cuda_major_ver, cuda_minor_ver = installed_cuda_version()
        if (cuda_major_ver < 11) or ((cuda_major_ver == 11) and (cuda_minor_ver < 3)):
            cupy = f"cupy-cuda{cuda_major_ver}{cuda_minor_ver}"
        else:
            cupy = f"cupy-cuda{cuda_major_ver}x"

    if cupy:
        extras_require['1bit'].append(cupy)
        extras_require['1bit_mpi'].append(cupy)

# Make an [all] extra that installs all needed dependencies.
all_extras = set()
for extra in extras_require.items():
    for req in extra[1]:
        all_extras.add(req)
extras_require['all'] = list(all_extras)

cmdclass = {}
cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if not torch.cuda.is_available():
    # Fix to allow docker buils, similar to https://github.com/NVIDIA/apex/issues/486
    print(
        "[WARNING] Torch did not find cuda available, if cross-compling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)")
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

# Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

ext_modules.append(
    CUDAExtension(name='fused_lamb_cuda',
                  sources=['csrc/fused_lamb_cuda.cpp',
                           'csrc/fused_lamb_cuda_kernel.cu'],
                  extra_compile_args={
                      'cxx': [
                          '-O3',
                      ] + version_dependent_macros,
                      'nvcc': ['-O3',
                               '--use_fast_math'] + version_dependent_macros
                  }))

setup(name='deepspeed',
      version=ds_version,
      description='DeepSpeed library',
      long_description=readme_text,
      long_description_content_type='text/markdown',
      author='DeepSpeed Team',
      author_email='deepspeed@microsoft.com',
      url='http://deepspeed.ai',
      install_requires=install_requires,
      packages=find_packages(exclude=["docker",
                                      "third_party",
                                      "csrc"]),
      scripts=['bin/deepspeed',
               'bin/deepspeed.pt',
               'bin/ds',
               'bin/ds_ssh'],
      classifiers=['Programming Language :: Python :: 3.6'],
      ext_modules=ext_modules,
      cmdclass=cmdclass)

end_time = time.time()
print(f'deepspeed build time = {end_time - start_time} secs')
