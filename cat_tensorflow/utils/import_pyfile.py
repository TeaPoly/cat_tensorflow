#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os


def import_pyfile(path, key=''):
    if not os.path.exists(path):
        raise Exception('File \"%s\" is not exists.' %(path))
    spec = importlib.util.spec_from_file_location(
        key, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
