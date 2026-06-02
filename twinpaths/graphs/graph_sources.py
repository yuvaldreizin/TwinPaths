import os
import pickle
from typing import Iterable, List, Tuple, Any

import networkx as nx


def _load_graph_file(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def discover_graph_paths(graph_dir: str, explicit: Iterable[str]) -> List[str]:
    """
    Return absolute paths to graph files to process.
    If explicit paths are provided, resolve each relative to CWD (not graph_dir).
    """
    if explicit:
        paths: List[str] = []
        for p in explicit:
            paths.append(os.path.abspath(p))
        return paths

    paths: List[str] = []
    for name in os.listdir(graph_dir):
        if name.lower().endswith(".pkl"):
            paths.append(os.path.abspath(os.path.join(graph_dir, name)))
    paths.sort()
    return paths


def load_graphs(graph_dir: str = "data/example_graphs", graphs: Iterable[str] = None) -> List[Tuple[str, nx.Graph]]:
    """
    Load graphs as networkx.Graph objects.
    If 'graphs' is provided, treat entries as file paths; otherwise, discover .pkl files in graph_dir.
    Returns list of (name, graph).
    """
    graph_paths = discover_graph_paths(graph_dir, graphs or [])
    loaded: List[Tuple[str, nx.Graph]] = []
    for path in graph_paths:
        g = _load_graph_file(path)
        loaded.append((os.path.basename(path), g))
    return loaded

