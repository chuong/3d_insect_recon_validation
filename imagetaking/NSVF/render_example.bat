:: Copyright (c) Facebook, Inc. and its affiliates.
::
:: This source code is licensed under the MIT license found in the
:: LICENSE file in the root directory of this source tree.

set BLENDER="E:\\Synthetic_NSVF\\ac\\ac_nsvf.blend"

set OUTPUT="E:\\Synthetic_NSVF\\ac\\ac_out"

blender --background %BLENDER% --python E:\Synthetic_NSVF\ac\render.py -- %OUTPUT%

pause