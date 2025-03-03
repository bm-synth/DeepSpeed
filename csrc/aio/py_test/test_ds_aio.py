"""
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import multiprocessing as mp
from ds_aio_basic import aio_basic_multiprocessing
from ds_aio_handle import aio_handle_multiprocessing
from ds_aio_args import get_validated_args


def main():
    print(f'Testing deepspeed_aio python frontend')

    args = get_validated_args()
    mp.set_start_method('spawn')
    multiprocess_function = aio_handle_multiprocessing if args.handle else aio_basic_multiprocessing
    multiprocess_function(args, args.read)


if __name__ == "__main__":
    main()
