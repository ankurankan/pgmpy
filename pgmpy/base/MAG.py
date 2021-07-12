#!/usr/bin/env python3

import networkx as nx

from pgmpy.base import DAG


class MixedGraph(nx.MultiDiGraph):
    """
    Class representing a Mixed Graph. A mixed graph can contain both a directed
    edge and a bi-directed edge. Bi-directed edges are represented using two
    edges between the same nodes in opposite directions. All the operations are
    although done on a canonical representation of the Mixed Graph which is a
    DAG. The canonical representation replaces each bidirected edge with a
    latent variable.  For example: A <-> B is converted to A <- _e_AB -> B.
    """

    def __init__(self, ebunch=None, latents=set()):
        """
        Initialzies a Mixed Graph.

        Parameters
        ----------

        ebunch: input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object. For bidirected edges e.g. X <-> Y, two
            edges need to be specified: X -> Y and Y <- X.

        latents: set, array-like
            List of variables which are latent (i.e. unobserved) in the model.
        """
        # TODO: Check why init is not taking the arguments directly.
        super(MixedGraph, self).__init__()
        if ebunch is None:
            ebunch = []
        self.add_edges_from(ebunch)
        self.latents = set(latents)

    def copy(self):
        """
        Returns a copy of the current object.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph([("A", "B"), ("A", "B"), ("B", "A")])
        >>> G_copy = G.copy()
        """
        return MixedGraph(ebunch=self.edges(), latents=self.latents)

    def to_canonical(self):
        """
        Converts a mixed graph into it's canonical representation.
        For each bi-directed edge, a latent variable is added as.
        For example: A <-> B is converted to A <- _e_AB -> B.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")])
        >>> G_canonical = G.to_canonical()
        >>> G_canonical.nodes()
        >>> G_canonical.edges()
        """
        bidirected_edges = []
        for u, v in self.edges():
            if (v, u) in self.edges():
                bidirected_edges.append(tuple(sorted([u, v])))

        G = self.copy()
        new_latents = set()
        for u, v in set(bidirected_edges):
            latent_node = f"_e_{str(u)}{str(v)}"
            new_latents.add(latent_node)

            G.add_edges_from([(latent_node, u), (latent_node, v)])
            G.remove_edges_from([(u, v), (v, u)])

        return DAG(ebunch=G.edges(), latents=self.latents.union(new_latents))

    def add_node(self, node, weight=None):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_node(node='A')
        >>> sorted(G.nodes())
        ['A']

        Adding a node with some weight.
        >>> G.add_node(node='B', weight=0.3)

        The weight of these nodes can be accessed as:
        >>> G.nodes['B']
        {'weight': 0.3}
        >>> G.nodes['A']
        {'weight': None}
        """
        super(MixedGraph, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None):
        """
        Add multiple nodes to the Graph.

        **The behviour of adding weights is different than in networkx.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))

        Adding nodes with weights:
        >>> G.add_nodes_from(nodes=['D', 'E'], weights=[0.3, 0.6])
        >>> G.nodes['D']
        {'weight': 0.3}
        >>> G.nodes['E']
        {'weight': 0.6}
        >>> G.nodes['A']
        {'weight': None}
        """
        nodes = list(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], weight=weights[index])
        else:
            for node in nodes:
                self.add_node(node=node)

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:
        >>> G.add_edge(u='Alice', v='Ankur')
        >>> G.nodes()
        NodeView(('Alice', 'Ankur', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Alice', 'Ankur')])

        Adding edges with weight:
        >>> G.add_edge('Ankur', 'Maria', weight=0.1)
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.1}
        """
        super(MixedGraph, self).add_edge(u, v, weight=weight)

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:
        >>> G.add_edges_from(ebunch=[('Alice', 'Ankur')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles', 'Ankur'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')])

        Adding edges with weights:
        >>> G.add_edges_from([('Ankur', 'Maria'), ('Maria', 'Mason')],
        ...                  weights=[0.3, 0.5])
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.3}
        >>> G.edge['Maria']['Mason']
        {'weight': 0.5}
        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError(
                    "The number of elements in ebunch and weights" "should be equal"
                )
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1], weight=weights[index])
        else:
            for edge in ebunch:
                self.add_edge(edge[0], edge[1])

    def get_spouse(self, node):
        """
        Returns the spouse of `node`. Spouse in mixed graph is defined as
        the nodes connected to `node` through a bi-directed edge.

        Parameters
        ----------
        node: any hashable python object
            The node whose spouses are needed.

        Returns
        -------
        list: List of spouses of `node`.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> g = MixedGraph([('X', 'Y'), ('Y', 'Z'), ('X', 'Z'), ('Z', 'X')])
        >>> g.get_spouse('X')
        ['Z']
        >>> g.get_spouse('Y')
        []
        """
        spouses = []
        for neigh in self.neighbors(node):
            if node in self.neighbors(neigh):
                spouses.append(neigh)

        return spouses


class MAG(MixedGraph):
    """
    Class representing Maximal Ancestral Graph (MAG)[1].

    References
    ----------
    [1] Zhang, Jiji. "Causal reasoning with ancestral graphs." Journal of Machine Learning Research 9 (2008): 1437-1474.
    """

    def __init__(self, ebunch=None, latents=set()):
        super(MAG, self).__init__(ebunch=directed_ebunch, latents=latents)
        pass


class PAG(MixedGraph):
    """
    Class representing Partial Ancestral Graph (PAG)[1].

    References
    ----------
    [1] Zhang, Jiji. "Causal reasoning with ancestral graphs." Journal of Machine Learning Research 9 (2008): 1437-1474.
    """

    def __init__(self, directed_ebunch=None, undirected_ebunch=None, latents=set()):
        super(PAG, self).__init__(ebunch=directed_ebunch, latents=latents)
        self.undirected_edges = set(undirected_ebunch)
