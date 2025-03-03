# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed library

To build wheel on Windows:
    1. Install pytorch, such as pytorch 1.8 + cuda 11.1
    2. Install visual cpp build tool
    3. Launch cmd console with Administrator privilege for creating required symlink folders

Create a new wheel via the following command:
build_win.bat

The wheel will be located at: dist/*.whl
"""

import os
import shutil
import subprocess
import warnings
import cpufeature
from setuptools import setup, find_packages

torch_available = True
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

from op_builder import ALL_OPS, get_default_compute_capatabilities


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
if torch_available and torch.cuda.is_available():
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

# For any pre-installed ops force disable ninja
if torch_available:
    cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

if torch_available:
    TORCH_MAJOR = torch.__version__.split('.')[0]
    TORCH_MINOR = torch.__version__.split('.')[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

if torch_available and not torch.cuda.is_available():
    # Fix to allow docker builds, similar to https://github.com/NVIDIA/apex/issues/486
    print(
        "[WARNING] Torch did not find cuda available, if cross-compiling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)")
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_default_compute_capatabilities()

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

# Default to pre-install kernels to false so we rely on JIT
BUILD_OP_DEFAULT = int(os.environ.get('DS_BUILD_OPS', 0))
print(f"DS_BUILD_OPS={BUILD_OP_DEFAULT}")

if BUILD_OP_DEFAULT:
    assert torch_available, "Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops."


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

install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()

    # If op is compatible update install reqs so it can potentially build/run later
    if op_compatible:
        reqs = builder.python_requirements()
        install_requires += builder.python_requirements()

    # If op install enabled, add builder to extensions
    if op_enabled(op_name) and op_compatible:
        assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
        install_ops[op_name] = op_enabled(op_name)
        ext_modules.append(builder.builder())

compatible_ops = {op_name: op.is_compatible() for (op_name, op) in ALL_OPS.items()}

print(f'Install Ops={install_ops}')

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


def create_dir_symlink(src, dest):
    if not os.path.islink(dest):
        if os.path.exists(dest):
            os.remove(dest)
        assert not os.path.exists(dest)
        os.symlink(src, dest)


if sys.platform == "win32":
    # This creates a symbolic links on Windows.
    # It needs Administrator privilege to create symlinks on Windows.
    create_dir_symlink('..\\..\\csrc', '.\\deepspeed\\ops\\csrc')
    create_dir_symlink('..\\..\\op_builder', '.\\deepspeed\\ops\\op_builder')

# Parse the DeepSpeed version string from version.txt
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# example: DS_BUILD_STR=".dev20201022" python setup.py sdist bdist_wheel
#version_str += os.environ.get('DS_BUILD_STRING', f'+{git_hash}')

# Building wheel for distribution, update version file

if 'DS_BUILD_STRING' in os.environ:
    # Build string env specified, probably building for distribution
    with open('build.txt', 'w') as fd:
        fd.write(os.environ.get('DS_BUILD_STRING'))
    version_str += os.environ.get('DS_BUILD_STRING')
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source
    version_str += f'+{git_hash}'

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
# Set cuda_version to 0.0 if cpu-only
cuda_version = "0.0"
if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
torch_info = {"version": torch_version, "cuda_version": cuda_version}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open('deepspeed/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{version_str}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")

print(f'install_requires={install_requires}')

# Parse README.md to make long_description for PyPI page.
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
    readme_text = fin.read()

start_time = time.time()

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
                                      "third_party"]),
      include_package_data=True,
      scripts=[
          'bin/deepspeed',
          'bin/deepspeed.pt',
          'bin/ds',
          'bin/ds_ssh',
          'bin/ds_report',
          'bin/ds_elastic'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      license='MIT',
      ext_modules=ext_modules,
      cmdclass=cmdclass)

end_time = time.time()
print(f'deepspeed build time = {end_time - start_time} secs')
