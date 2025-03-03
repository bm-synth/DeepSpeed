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
import shutil
import subprocess
import warnings
import cpufeature
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

# If MPI is available add 1bit-adam requirements
if torch.cuda.is_available():
    if shutil.which('ompi_info') or shutil.which('mpiname'):
        onebit_adam_requires = fetch_requirements(
            'requirements/requirements-1bit-adam.txt')
        onebit_adam_requires.append(f"cupy-cuda{torch.version.cuda.replace('.','')[:3]}")
        install_requires += onebit_adam_requires

# Constants for each op
LAMB = "lamb"
TRANSFORMER = "transformer"
SPARSE_ATTN = "sparse-attn"
CPU_ADAM = "cpu-adam"

# Build environment variables for custom builds
DS_BUILD_LAMB_MASK = 1
DS_BUILD_TRANSFORMER_MASK = 10
DS_BUILD_SPARSE_ATTN_MASK = 100
DS_BUILD_CPU_ADAM_MASK = 1000
DS_BUILD_AVX512_MASK = 10000

# Allow for build_cuda to turn on or off all ops
DS_BUILD_ALL_OPS = DS_BUILD_LAMB_MASK | DS_BUILD_TRANSFORMER_MASK | DS_BUILD_SPARSE_ATTN_MASK | DS_BUILD_CPU_ADAM_MASK | DS_BUILD_AVX512_MASK
DS_BUILD_CUDA = int(os.environ.get('DS_BUILD_CUDA', 1)) * DS_BUILD_ALL_OPS

# Set default of each op based on if build_cuda is set
OP_DEFAULT = DS_BUILD_CUDA == DS_BUILD_ALL_OPS
DS_BUILD_CPU_ADAM = int(os.environ.get('DS_BUILD_CPU_ADAM',
                                       OP_DEFAULT)) * DS_BUILD_CPU_ADAM_MASK
DS_BUILD_LAMB = int(os.environ.get('DS_BUILD_LAMB', OP_DEFAULT)) * DS_BUILD_LAMB_MASK
DS_BUILD_TRANSFORMER = int(os.environ.get('DS_BUILD_TRANSFORMER',
                                          OP_DEFAULT)) * DS_BUILD_TRANSFORMER_MASK
DS_BUILD_SPARSE_ATTN = int(os.environ.get('DS_BUILD_SPARSE_ATTN',
                                          OP_DEFAULT)) * DS_BUILD_SPARSE_ATTN_MASK
DS_BUILD_AVX512 = int(os.environ.get(
    'DS_BUILD_AVX512',
    cpufeature.CPUFeature['AVX512f'])) * DS_BUILD_AVX512_MASK

# Final effective mask is the bitwise OR of each op
BUILD_MASK = (DS_BUILD_LAMB | DS_BUILD_TRANSFORMER | DS_BUILD_SPARSE_ATTN
              | DS_BUILD_CPU_ADAM)

install_ops = dict.fromkeys([LAMB, TRANSFORMER, SPARSE_ATTN, CPU_ADAM], False)
if BUILD_MASK & DS_BUILD_LAMB:
    install_ops[LAMB] = True
if BUILD_MASK & DS_BUILD_CPU_ADAM:
    install_ops[CPU_ADAM] = True
if BUILD_MASK & DS_BUILD_TRANSFORMER:
    install_ops[TRANSFORMER] = True
if BUILD_MASK & DS_BUILD_SPARSE_ATTN:
    install_ops[SPARSE_ATTN] = True
if len(install_ops) == 0:
    print("Building without any cuda/cpp extensions")
print(f'BUILD_MASK={BUILD_MASK}, install_ops={install_ops}')

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

cpu_info = cpufeature.CPUFeature
SIMD_WIDTH = ''
if cpu_info['AVX512f'] and DS_BUILD_AVX512:
    SIMD_WIDTH = '-D__AVX512__'
elif cpu_info['AVX2']:
    SIMD_WIDTH = '-D__AVX256__'
print("SIMD_WIDTH = ", SIMD_WIDTH)

ext_modules = []

## Lamb ##
if BUILD_MASK & DS_BUILD_LAMB:
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.lamb.fused_lamb_cuda',
                      sources=[
                          'csrc/lamb/fused_lamb_cuda.cpp',
                          'csrc/lamb/fused_lamb_cuda_kernel.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': [
                              '-O3',
                          ] + version_dependent_macros,
                          'nvcc': ['-O3',
                                   '--use_fast_math'] + version_dependent_macros
                      }))

## Adam ##
if BUILD_MASK & DS_BUILD_CPU_ADAM:
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.adam.cpu_adam_op',
                      sources=[
                          'csrc/adam/cpu_adam.cpp',
                          'csrc/adam/custom_cuda_kernel.cu',
                      ],
                      include_dirs=['csrc/includes',
                                    '/usr/local/cuda/include'],
                      extra_compile_args={
                          'cxx': [
                              '-O3',
                              '-std=c++14',
                              '-L/usr/local/cuda/lib64',
                              '-lcudart',
                              '-lcublas',
                              '-g',
                              '-Wno-reorder',
                              '-march=native',
                              '-fopenmp',
                              SIMD_WIDTH
                          ],
                          'nvcc': [
                              '-O3',
                              '--use_fast_math',
                              '-gencode',
                              'arch=compute_61,code=compute_61',
                              '-gencode',
                              'arch=compute_70,code=compute_70',
                              '-std=c++14',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-U__CUDA_NO_HALF2_OPERATORS__'
                          ]
                      }))

## Transformer ##
if BUILD_MASK & DS_BUILD_TRANSFORMER:
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.transformer.transformer_cuda',
                      sources=[
                          'csrc/transformer/ds_transformer_cuda.cpp',
                          'csrc/transformer/cublas_wrappers.cu',
                          'csrc/transformer/transform_kernels.cu',
                          'csrc/transformer/gelu_kernels.cu',
                          'csrc/transformer/dropout_kernels.cu',
                          'csrc/transformer/normalize_kernels.cu',
                          'csrc/transformer/softmax_kernels.cu',
                          'csrc/transformer/general_kernels.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': ['-O3',
                                  '-std=c++14',
                                  '-g',
                                  '-Wno-reorder'],
                          'nvcc': [
                              '-O3',
                              '--use_fast_math',
                              '-gencode',
                              'arch=compute_61,code=compute_61',
                              '-gencode',
                              'arch=compute_60,code=compute_60',
                              '-gencode',
                              'arch=compute_70,code=compute_70',
                              '-std=c++14',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-U__CUDA_NO_HALF2_OPERATORS__'
                          ]
                      }))
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.transformer.stochastic_transformer_cuda',
                      sources=[
                          'csrc/transformer/ds_transformer_cuda.cpp',
                          'csrc/transformer/cublas_wrappers.cu',
                          'csrc/transformer/transform_kernels.cu',
                          'csrc/transformer/gelu_kernels.cu',
                          'csrc/transformer/dropout_kernels.cu',
                          'csrc/transformer/normalize_kernels.cu',
                          'csrc/transformer/softmax_kernels.cu',
                          'csrc/transformer/general_kernels.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': ['-O3',
                                  '-std=c++14',
                                  '-g',
                                  '-Wno-reorder'],
                          'nvcc': [
                              '-O3',
                              '--use_fast_math',
                              '-gencode',
                              'arch=compute_61,code=compute_61',
                              '-gencode',
                              'arch=compute_60,code=compute_60',
                              '-gencode',
                              'arch=compute_70,code=compute_70',
                              '-std=c++14',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-U__CUDA_NO_HALF2_OPERATORS__',
                              '-D__STOCHASTIC_MODE__'
                          ]
                      }))


def command_exists(cmd):
    if '|' in cmd:
        cmds = cmd.split("|")
    else:
        cmds = [cmd]
    valid = False
    for cmd in cmds:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        valid = valid or result.wait() == 0
    return valid


## Sparse transformer ##
if BUILD_MASK & DS_BUILD_SPARSE_ATTN:
    # Check to see if llvm and cmake are installed since they are dependencies
    required_commands = ['llvm-config|llvm-config-9', 'cmake']

    command_status = list(map(command_exists, required_commands))
    if not all(command_status):
        zipped_status = list(zip(required_commands, command_status))
        warnings.warn(
            f'Missing non-python requirements, please install the missing packages: {zipped_status}'
        )
        warnings.warn(
            'Skipping sparse attention installation due to missing required packages')
        # remove from installed ops list
        install_ops[SPARSE_ATTN] = False
    elif TORCH_MAJOR == 1 and TORCH_MINOR >= 5:
        ext_modules.append(
            CppExtension(name='deepspeed.ops.sparse_attention.cpp_utils',
                         sources=['csrc/sparse_attention/utils.cpp'],
                         extra_compile_args={'cxx': ['-O2',
                                                     '-fopenmp']}))
        # Add sparse attention requirements
        install_requires += sparse_attn_requires
    else:
        warnings.warn('Unable to meet requirements to install sparse attention')
        # remove from installed ops list
        install_ops[SPARSE_ATTN] = False

# Add development requirements
install_requires += dev_requires

# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git'):
    result = subprocess.check_output(git_hash_cmd, shell=True)
    git_hash = result.decode('utf-8').strip()
    result = subprocess.check_output(git_branch_cmd, shell=True)
    git_branch = result.decode('utf-8').strip()
else:
    git_hash = "unknown"
    git_branch = "unknown"
print(f"version={VERSION}+{git_hash}, git_hash={git_hash}, git_branch={git_branch}")
with open('deepspeed/git_version_info.py', 'w') as fd:
    fd.write(f"version='{VERSION}+{git_hash}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")

print(f'install_requires={install_requires}')

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
