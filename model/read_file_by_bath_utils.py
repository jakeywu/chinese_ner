#!/usr/bin/env python3
# encoding: utf-8

import copy


class AA(object):

    def __init__(self, bathsize, split_flag):
        self._bathsize = bathsize
        self._split_flag = split_flag
        self._data = self.prepare_data()

    @staticmethod
    def prepare_data():
        with open("./trainset", "r") as fp:
            while True:
                a_line = fp.readline()
                if a_line:
                    yield a_line.strip()
                else:
                    break

    def __next__(self):
        _2d = []
        first = self._bathsize
        try:
            _1d = []
            while first > 0:

                cur = next(self._data)
                if not cur.endswith(self._split_flag):
                    _1d.append(cur)
                else:
                    first -= 1
                    _2d.append(copy.deepcopy(_1d))
                    _1d = []

        except StopIteration as e:
            if first >= self._bathsize:
                raise StopIteration()

        return _2d

# AA.prepare_data()
a = AA(10, "end")

while True:
    print(next(a))


