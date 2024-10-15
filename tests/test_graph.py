import unittest
from pathlib import Path

from aioxmpp import JID

from macofl.datatypes import GraphManager


class TestGraph(unittest.TestCase):

    def __init__(self, methodName="test_initialization") -> None:
        super().__init__(methodName)
        self.folder: Path = Path("premiofl_graphs")

    def test_initialization(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

    def test_custom_graph(self) -> None:
        gml_manager = GraphManager()

        agent1 = JID.fromstr("agent1@localhost")
        agent2 = JID.fromstr("agent2@localhost")
        agent3 = JID.fromstr("agent3@localhost")
        agent4 = JID.fromstr("agent4@localhost")
        agent5 = JID.fromstr("agent5@localhost")
        agent6 = JID.fromstr("agent6@localhost")

        # Adding agents and connections
        gml_manager.add_agent(agent1, coalition_id=1)
        gml_manager.add_agent(agent2, coalition_id=1)
        gml_manager.add_agent(agent3, coalition_id=2)
        gml_manager.add_agent(agent4, coalition_id=2)
        gml_manager.add_agent(agent5, coalition_id=3)
        gml_manager.add_agent(agent6, coalition_id=None)

        gml_manager.add_connection(agent1, agent2)
        gml_manager.add_connection(agent2, agent3)
        gml_manager.add_connection(agent3, agent4)
        gml_manager.add_connection(agent4, agent2)
        gml_manager.add_connection(agent5, agent4)
        gml_manager.add_connection(agent6, agent4)

        # Exporting to GML
        out = self.folder / "agents_graph"
        gml_manager.export_to_gml(f"{out}.gml")
        gml_manager.import_from_gml(f"{out}.gml")

        # Visualizing the graph
        gml_manager.visualize(f"{out}.html")

    def test_generated_graphs(self) -> None:
        gml_manager = GraphManager()
        agents = [JID.fromstr(f"agent{i}@localhost") for i in range(10)]

        # Generate a ring structure
        out = self.folder / "agents_ring"
        out.resolve()
        gml_manager.generate_ring(agents)
        gml_manager.export_to_gml(f"{out}.gml")
        gml_manager.import_from_gml(f"{out}.gml")
        gml_manager.visualize(f"{out}.html")

        # Complete graph
        out = self.folder / "agents_complete"
        out.resolve()
        gml_manager.generate_complete(agents)
        gml_manager.export_to_gml(f"{out}.gml")
        gml_manager.import_from_gml(f"{out}.gml")
        gml_manager.visualize(f"{out}.html")

        # Small-world graph
        out = self.folder / "agents_sw"
        out.resolve()
        gml_manager.generate_small_world(agents, k=2, p=0.1)
        gml_manager.export_to_gml(f"{out}.gml")
        gml_manager.import_from_gml(f"{out}.gml")
        gml_manager.visualize(f"{out}.html")
