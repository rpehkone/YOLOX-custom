#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# import torch first to make jit op work without `ImportError of libc10.so`
import torch  # noqa

from .jit_ops import JitOp
