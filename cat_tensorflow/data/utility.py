#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools


def lines(path):
    return sum(1 for line in open(path))


def read_scp(path, start=None, end=None):
    def parse_scp(line):
        line_split = line.strip().split()
        uuid = line_split[0]
        params = line_split[1].split(':')
        wav = params[0]
        #  wav = line_split[1]
        return uuid, wav

    with open(path, 'r') as fp:
        if start is None and end is None:
            for line in fp:
                uuid, wav = parse_scp(line)
                yield uuid, wav
        else:
            for line in itertools.islice(fp, start, end):
                uuid, wav = parse_scp(line)
                yield uuid, wav


def read_labels(path, start=None, end=None):
    def parse_lbl(line):
        line_split = line.strip().split()
        uuid = line_split[0]
        labels = list(map(int, line_split[1:]))
        return uuid, labels

    with open(path, 'r') as fp:
        for line in itertools.islice(fp, start, end):
            uuid, labels = parse_lbl(line)
            yield uuid, labels


def read_weight(path, start=None, end=None):
    def parse_weight(line):
        uuid, weight = line.strip().split()
        return uuid, float(weight)

    with open(path, 'r') as fp:
        for line in itertools.islice(fp, start, end):
            uuid, weight = parse_weight(line)
            yield uuid, weight

