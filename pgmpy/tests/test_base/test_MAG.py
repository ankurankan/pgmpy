import unittest

from pgmpy.base import MixedGraph, MAG, PAG


class TestMixedGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = MixedGraph()

    def test_only_directed(self):
        # Graph 1: Without latents
        G = MixedGraph([("A", "B"), ("B", "C")])
        self.assertEqual(set(G.nodes()), set(["A", "B", "C"]))
        self.assertEqual(set(G.edges()), set([("A", "B"), ("B", "C")]))
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set(["A", "B", "C"]))
        self.assertEqual(set(G_canonical.edges()), set([("A", "B"), ("B", "C")]))
        self.assertEqual(G_canonical.latents, set())

        # Graph 1: With latents
        G = MixedGraph([("A", "B"), ("B", "C")], latents=["B"])
        self.assertEqual(set(G.nodes()), set(["A", "B", "C"]))
        self.assertEqual(set(G.edges()), set([("A", "B"), ("B", "C")]))
        self.assertEqual(G.latents, set(["B"]))

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set(["A", "B", "C"]))
        self.assertEqual(set(G_canonical.edges()), set([("A", "B"), ("B", "C")]))
        self.assertEqual(G_canonical.latents, set(["B"]))

        # Graph 2: Without latents
        G = MixedGraph([(1, 2), (2, 3)])
        self.assertEqual(set(G.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G.edges()), set([(1, 2), (2, 3)]))
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G_canonical.edges()), set([(1, 2), (2, 3)]))
        self.assertEqual(G_canonical.latents, set())

        # Graph 2: With latents
        G = MixedGraph([(1, 2), (2, 3)], latents=set([2]))
        self.assertEqual(set(G.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G.edges()), set([(1, 2), (2, 3)]))
        self.assertEqual(G.latents, set([2]))

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G_canonical.edges()), set([(1, 2), (2, 3)]))
        self.assertEqual(G_canonical.latents, set([2]))

        # Graph 3: Without latents
        G = MixedGraph([((1, 2), (3, 4)), ((3, 4), (5, 6))])
        self.assertEqual(set(G.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(set(G.edges()), set([((1, 2), (3, 4)), ((3, 4), (5, 6))]))
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(
            set(G_canonical.edges()), set([((1, 2), (3, 4)), ((3, 4), (5, 6))])
        )
        self.assertEqual(G_canonical.latents, set())

        # Graph 3: With latents
        G = MixedGraph(
            [((1, 2), (3, 4)), ((3, 4), (5, 6))],
            latents=set(
                [
                    (3, 4),
                ]
            ),
        )
        self.assertEqual(set(G.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(set(G.edges()), set([((1, 2), (3, 4)), ((3, 4), (5, 6))]))
        self.assertEqual(
            G.latents,
            set(
                [
                    (3, 4),
                ]
            ),
        )

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(
            set(G_canonical.edges()), set([((1, 2), (3, 4)), ((3, 4), (5, 6))])
        )
        self.assertEqual(
            G_canonical.latents,
            set(
                [
                    (3, 4),
                ]
            ),
        )

    def test_bidirected(self):
        # Graph 1: Without latents
        G = MixedGraph([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")])
        self.assertEqual(set(G.nodes()), set(["A", "B", "C"]))
        self.assertEqual(
            set(G.edges()), set([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")])
        )
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set(["A", "B", "C", "_e_AC"]))
        self.assertEqual(
            set(G_canonical.edges()),
            set([("A", "B"), ("B", "C"), ("_e_AC", "A"), ("_e_AC", "C")]),
        )
        self.assertEqual(
            G_canonical.latents,
            set(
                [
                    "_e_AC",
                ]
            ),
        )

        # Graph 1: With latents
        G = MixedGraph([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")], latents=["B"])
        self.assertEqual(set(G.nodes()), set(["A", "B", "C"]))
        self.assertEqual(
            set(G.edges()), set([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")])
        )
        self.assertEqual(
            G.latents,
            set(
                [
                    "B",
                ]
            ),
        )
        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set(["A", "B", "C", "_e_AC"]))
        self.assertEqual(
            set(G_canonical.edges()),
            set([("A", "B"), ("B", "C"), ("_e_AC", "A"), ("_e_AC", "C")]),
        )
        self.assertEqual(G_canonical.latents, set(["_e_AC", "B"]))

        # Graph 2: Without latents
        G = MixedGraph([(1, 2), (2, 3), (1, 3), (3, 1)])
        self.assertEqual(set(G.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G.edges()), set([(1, 2), (2, 3), (1, 3), (3, 1)]))
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([1, 2, 3, "_e_13"]))
        self.assertEqual(
            set(G_canonical.edges()), set([(1, 2), (2, 3), ("_e_13", 1), ("_e_13", 3)])
        )
        self.assertEqual(
            G_canonical.latents,
            set(
                [
                    "_e_13",
                ]
            ),
        )

        # Graph 2: With latents
        G = MixedGraph([(1, 2), (2, 3), (1, 3), (3, 1)], latents=set([2]))
        self.assertEqual(set(G.nodes()), set([1, 2, 3]))
        self.assertEqual(set(G.edges()), set([(1, 2), (2, 3), (3, 1), (1, 3)]))
        self.assertEqual(G.latents, set([2]))

        G_canonical = G.to_canonical()
        self.assertEqual(set(G_canonical.nodes()), set([1, 2, 3, "_e_13"]))
        self.assertEqual(
            set(G_canonical.edges()), set([(1, 2), (2, 3), ("_e_13", 1), ("_e_13", 3)])
        )
        self.assertEqual(G_canonical.latents, set(["_e_13", 2]))

        # Graph 3: Without latents
        G = MixedGraph(
            [((1, 2), (3, 4)), ((3, 4), (5, 6)), ((1, 2), (5, 6)), ((5, 6), (1, 2))]
        )
        self.assertEqual(set(G.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(
            set(G.edges()),
            set(
                [((1, 2), (3, 4)), ((3, 4), (5, 6)), ((1, 2), (5, 6)), ((5, 6), (1, 2))]
            ),
        )
        self.assertEqual(G.latents, set())

        G_canonical = G.to_canonical()
        self.assertEqual(
            set(G_canonical.nodes()), set([(1, 2), (3, 4), (5, 6), "_e_(1, 2)(5, 6)"])
        )
        self.assertEqual(
            set(G_canonical.edges()),
            set(
                [
                    ((1, 2), (3, 4)),
                    ((3, 4), (5, 6)),
                    ("_e_(1, 2)(5, 6)", (1, 2)),
                    ("_e_(1, 2)(5, 6)", (5, 6)),
                ]
            ),
        )
        self.assertEqual(
            G_canonical.latents,
            set(
                [
                    "_e_(1, 2)(5, 6)",
                ]
            ),
        )

        # Graph 3: With latents
        G = MixedGraph(
            [((1, 2), (3, 4)), ((3, 4), (5, 6)), ((1, 2), (5, 6)), ((5, 6), (1, 2))],
            latents=[
                (3, 4),
            ],
        )
        self.assertEqual(set(G.nodes()), set([(1, 2), (3, 4), (5, 6)]))
        self.assertEqual(
            set(G.edges()),
            set(
                [((1, 2), (3, 4)), ((3, 4), (5, 6)), ((1, 2), (5, 6)), ((5, 6), (1, 2))]
            ),
        )
        self.assertEqual(
            G.latents,
            set(
                [
                    (3, 4),
                ]
            ),
        )

        G_canonical = G.to_canonical()
        self.assertEqual(
            set(G_canonical.nodes()), set([(1, 2), (3, 4), (5, 6), "_e_(1, 2)(5, 6)"])
        )
        self.assertEqual(
            set(G_canonical.edges()),
            set(
                [
                    ((1, 2), (3, 4)),
                    ((3, 4), (5, 6)),
                    ("_e_(1, 2)(5, 6)", (1, 2)),
                    ("_e_(1, 2)(5, 6)", (5, 6)),
                ]
            ),
        )
        self.assertEqual(G_canonical.latents, set(["_e_(1, 2)(5, 6)", (3, 4)]))

    def test_both_dir_bidir(self):
        G = MixedGraph([("X", "Y"), ("X", "Y"), ("Y", "X")])
        self.assertEqual(G.number_of_edges(), 3)
        self.assertEqual(G.number_of_nodes(), 2)

        G_canonical = G.to_canonical()
        self.assertEqual(
            set(G_canonical.edges()), set([("X", "Y"), ("_e_XY", "X"), ("_e_XY", "Y")])
        )
        self.assertEqual(G_canonical.latents, set(["_e_XY"]))
        self.assertEqual(G_canonical.number_of_edges(), 3)
        self.assertEqual(G_canonical.number_of_nodes(), 3)


class TestMixedGraphMethod(unittest.TestCase):
    def setUp(self):
        self.model1 = MixedGraph([("X", "Y"), ("Y", "Z"), ("X", "Z"), ("Z", "X")])
        self.model2 = MixedGraph(
            [("X", "Y"), ("Y", "Z"), ("X", "Z"), ("Z", "X"), ("Y", "Z"), ("Z", "Y")]
        )

    def test_get_spouse(self):
        self.assertEqual(self.model1.get_spouse("X"), ["Z"])
        self.assertEqual(self.model1.get_spouse("Y"), [])
        self.assertEqual(self.model1.get_spouse("Z"), ["X"])

        self.assertEqual(self.model2.get_spouse("X"), ["Z"])
        self.assertEqual(self.model2.get_spouse("Y"), ["Z"])
        self.assertEqual(sorted(self.model2.get_spouse("Z")), ["X", "Y"])