# Copyright 2024 The OpenXLA Authors.
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
# ==============================================================================
"""Boilerplate utilities for cpu kernel testing."""


from xla.backends.cpu import testlib as testlib_cpu
from xla.codegen import testlib as testlib_base


def annotate_hlo_module(
    hlo_module: testlib_base.HloModule,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  hlo_compiler = testlib_cpu.HloCompiler()
  hlo_module.set_schedule(hlo_compiler.create_hlo_schedule(hlo_module))
  buffer_assignment = hlo_compiler.create_buffer_assignment(hlo_module)
  return hlo_module, buffer_assignment


def parse_hlo_module(
    hlo_string: str,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  hlo_module = testlib_base.HloModule.parse_from_string(hlo_string)
  return annotate_hlo_module(hlo_module)


def build_hlo_module(
    root: testlib_base.HloInstruction,
    *instructions: testlib_base.HloInstruction,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  hlo_module = testlib_base.HloModule.build(root, *instructions)
  return annotate_hlo_module(hlo_module)