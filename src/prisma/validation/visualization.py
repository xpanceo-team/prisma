from dataclasses import dataclass, asdict
from typing import Iterable, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

from prisma.utils.logging import logger


@dataclass
class EHullEntryData:
    formula: str
    material_id: str
    e_above_hull_per_atom: float

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EHullData:
    """Prepared x/y points and hover-metadata for one class of entries."""

    name: str
    x: np.ndarray
    y: np.ndarray
    metadata: list[EHullEntryData]
    chemical_system: set[str]
    second_element: str


class EHull2DDataBuilder:
    """
    Computes (atomic‑fraction, E_hull_atom) pairs once
    and hands them to any Plotly/Matplotlib front end.
    """

    max_elements: int = 2

    def __init__(self, phase_diagram: PhaseDiagram, second_element: str) -> None:
        self._phase_diagram = phase_diagram
        self._second_element = second_element

    def build(self, entries: Iterable[PDEntry], name: str) -> EHullData:
        data = self._process_entries(entries, name)

        return data

    def _process_entries(self, entries: Iterable[PDEntry], name: str) -> EHullData:
        xs = []
        ys = []
        metadatas = []
        chemical_system = set()

        for entry in entries:
            chemical_system = chemical_system.union(
                [str(element) for element in entry.elements]
            )
            if len(chemical_system) > self.max_elements:
                raise ValueError(
                    f"Cannot create data for 2D E_Hull plot "
                    f"with more than {self.max_elements} elements."
                )

            x, y, metadata = self._process_entry(entry)

            xs.append(x)
            ys.append(y)
            metadatas.append(metadata)

        zipped_lists = zip(xs, ys, metadatas)
        sorted_zipped = sorted(zipped_lists, key=lambda lists: lists[0])
        xs, ys, metadatas = map(list, zip(*sorted_zipped))

        data = EHullData(
            name=name,
            x=np.array(xs),
            y=np.array(ys),
            metadata=metadatas,
            chemical_system=chemical_system,
            second_element=self._second_element,
        )

        return data

    def _process_entry(self, e: PDEntry):
        e_above_hull_total = self._phase_diagram.get_e_above_hull(e)
        e_above_hull_per_atom = e_above_hull_total
        ref_element_fraction = e.composition.get_atomic_fraction(self._second_element)

        metadata = EHullEntryData(
            formula=e.reduced_formula,
            material_id=e.attribute.get("material_id", "N/A"),
            e_above_hull_per_atom=e_above_hull_per_atom,
        )

        x = ref_element_fraction
        y = e_above_hull_per_atom

        return x, y, metadata


class Hull2DPlotter:
    """
    Draws convex hull with plotly.
    """

    max_elements: int = 2

    def __init__(self, title: str | None = None) -> None:
        self._second_element: Optional[str] = None
        self.title = title

        self._fig = go.Figure()
        self._chemical_system = set()

        self._saved_data: list[EHullData] = []

    def add(
        self,
        data: EHullData,
        color: str,
        marker: str,
        size: float = 5,
    ) -> None:
        chemical_system = self._chemical_system.union(data.chemical_system)
        if len(chemical_system) > self.max_elements:
            raise ValueError(
                f"Cannot add data with {data.chemical_system} system "
                f"for 2D E_Hull plot with {self._chemical_system} system."
                f"More than {self.max_elements} elements is not allowed."
            )

        if self._second_element is None:
            self._second_element = data.second_element
        elif self._second_element != data.second_element:
            raise ValueError(
                f"Trying to add data build with '{data.second_element}' "
                f"element fraction but expected "
                f"{self._second_element} as previously added."
            )

        self._fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode="markers",
                name=data.name,
                marker=dict(size=size, color=color, symbol=marker),
                customdata=[
                    (m.formula, m.material_id, m.e_above_hull_per_atom)
                    for m in data.metadata
                ],
                hovertemplate=(
                    "Formula: %{customdata[0]}<br>"
                    "ID: %{customdata[1]}<br>"
                    "E_above_hull/atom: %{customdata[2]:.4f} eV"
                ),
            )
        )

        self._saved_data.append(data)
        self._chemical_system = chemical_system

        logger.debug(
            f"Added {len(data.metadata)} entries to Hull2DPlotter as {data.name}"
        )

    def show(self, per_fraction: bool = True) -> None:
        if per_fraction:
            self._show_per_fraction()
        else:
            self._show_per_formula()

    def _show_per_fraction(self) -> None:
        title = self.title or f"Convex hull for {self._chemical_system} system"

        self._fig.update_layout(
            title=title,
            xaxis_title=f"Atomic fraction of {self._second_element}",
            yaxis_title="Energy Above Hull per Atom (eV)",
            legend_title="Legend",
        )
        self._fig.show()

    def _show_per_formula(self) -> None:
        df_data = []
        for data in self._saved_data:
            data_type = data.name
            xs = data.x

            for x, metadata in zip(xs, data.metadata):
                d = metadata.dict()
                d["type"] = data_type
                d["xs"] = x

                df_data.append(d)

        df = pd.DataFrame(df_data)
        df = df.sort_values("xs")
        df = df.drop(columns="xs")

        title = self.title or f"Convex hull for {self._chemical_system} system"

        fig = px.scatter(
            df,
            x="formula",
            y="e_above_hull_per_atom",
            color="type",
            title=title,
            labels={
                "formula": "Formula",
                "e_above_hull_per_atom": "E above hull / atom (eV)",
            },
        )

        fig.update_xaxes(categoryarray=df["formula"].drop_duplicates())

        fig.show()
