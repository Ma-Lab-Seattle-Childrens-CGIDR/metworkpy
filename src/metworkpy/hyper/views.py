"""
Classes for views of pieces of HyperGraphs
"""

from collections.abc import Mapping


class ReadOnlyDictView(Mapping):
    """A read only view of a dict"""

    __slots__ = ("_items",)

    def __getstate__(self):
        return {"_items": self._items}

    def __setstate__(self, state):
        self._items = state["_items"]

    def __init__(self, d):
        self._items = d

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def copy(self):
        return self._items.copy()

    def __str__(self):
        return str(self._items)  # {nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._items!r})"


# This is from NetworkX (networkx.classes.coreviews),
# licensed under the BSD-3-Clause (https://github.com/networkx/networkx/blob/main/LICENSE.txt),
# see the LICENSE file in this repository for the full license
class AtlasView(Mapping):
    """An AtlasView is a Read-only Mapping of Mappings.

    It is a View into a dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer level is read-only.

    See Also
    ========
    AdjacencyView: View into dict-of-dict-of-dict
    """

    __slots__ = ("_atlas",)

    def __getstate__(self):
        return {"_atlas": self._atlas}

    def __setstate__(self, state):
        self._atlas = state["_atlas"]

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)  # {nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r})"


# This is from NetworkX (networkx.classes.coreviews),
# licensed under the BSD-3-Clause (https://github.com/networkx/networkx/blob/main/LICENSE.txt),
# see the LICENSE file in this repository for the full license
class AdjacencyView(AtlasView):
    """An AdjacencyView is a Read-only Map of Maps of Maps.

    It is a View into a dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    """

    __slots__ = ()  # Still uses AtlasView slots names _atlas

    def __getitem__(self, key):
        return AtlasView(self._atlas[key])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}
