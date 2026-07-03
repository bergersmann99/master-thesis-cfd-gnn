"""
createGeometry.py
=================
Erzeugt ein freistehendes Satteldach-Haus als STL-Datei.

Das Haus wird als extrudiertes Profil (Rechteck + Giebel) modelliert.
Die Rotation um die Z-Achse simuliert verschiedene Windrichtungen,
waehrend der numerische Windkanal (Domain, Inlet-Profil) konstant bleibt.

Querschnitt (Y-Z Ebene):
        /\\
       /  \\
      /    \\
     +------+
     |      |
     |      |
     +------+

Referenzen:
    - numpy-stl Dokumentation: https://numpy-stl.readthedocs.io/
    - AIJ-Guideline Abschnitt 2.2 (Geometrie)
"""

import numpy as np
import math
import sys
import os

try:
    from stl import mesh as stl_mesh
except ImportError:
    print("FEHLER: 'numpy-stl' nicht installiert.")
    print("        Installation: pip install numpy-stl")
    sys.exit(1)


def create_building(width, depth, wall_height, roof_height, angle_deg, filepath):
    """
    Erzeugt ein Satteldach-Haus und speichert es als STL.

    Das Haus ist zentriert um den Ursprung (0, 0) mit Boden auf z=0.
    Bei angle_deg != 0 wird das Gebaeude um die Z-Achse rotiert.

    Parameter
    ---------
    width : float
        Gebaeude-Breite [m] (X-Richtung vor Rotation).
    depth : float
        Gebaeude-Tiefe [m] (Y-Richtung vor Rotation).
    wall_height : float
        Wandhoehe [m].
    roof_height : float
        Dachhoehe ueber Wandoberkante [m].
    angle_deg : float
        Rotationswinkel [Grad] um die Z-Achse.
    filepath : str
        Pfad fuer die STL-Ausgabedatei.

    Rueckgabe
    ---------
    tuple : (min_x, max_x, min_y, max_y, min_z, max_z)
        Bounding Box des rotierten Gebaeudes.
    """
    w2 = width / 2.0
    d2 = depth / 2.0
    hw = wall_height
    hr = roof_height

    # ------------------------------------------------------------------
    # Vertices: 10 Eckpunkte
    #   0-3: Boden
    #   4-7: Traufe (Wandoberkante)
    #   8-9: Dachfirst (Mitte oben, entlang Y-Achse)
    # ------------------------------------------------------------------
    vertices = np.array([
        [-w2, -d2, 0.0],       # 0  Boden vorne links
        [ w2, -d2, 0.0],       # 1  Boden vorne rechts
        [ w2,  d2, 0.0],       # 2  Boden hinten rechts
        [-w2,  d2, 0.0],       # 3  Boden hinten links
        [-w2, -d2, hw],        # 4  Traufe vorne links
        [ w2, -d2, hw],        # 5  Traufe vorne rechts
        [ w2,  d2, hw],        # 6  Traufe hinten rechts
        [-w2,  d2, hw],        # 7  Traufe hinten links
        [ 0.0, -d2, hw + hr],  # 8  First vorne
        [ 0.0,  d2, hw + hr],  # 9  First hinten
    ])

    # ------------------------------------------------------------------
    # Dreiecke (Faces): 14 Stueck
    #   - Boden:            2 Dreiecke  (1 Rechteck)
    #   - Laengswaende:     4 Dreiecke  (2 Rechtecke)
    #   - Giebelseiten:     4 Dreiecke  (2x Rechteck + Dreieck)
    #   - Dachflaechen:     4 Dreiecke  (2 Rechtecke)
    # ------------------------------------------------------------------
    faces = np.array([
        # Boden
        [0, 2, 1],
        [0, 3, 2],

        # Laengswand links  (-X Seite)
        [0, 4, 7],
        [0, 7, 3],

        # Laengswand rechts (+X Seite)
        [1, 2, 6],
        [1, 6, 5],

        # Giebelseite vorne (-Y Seite): Rechteck + Dreieck
        [0, 1, 5],
        [0, 5, 4],
        [4, 5, 8],

        # Giebelseite hinten (+Y Seite): Rechteck + Dreieck
        [3, 7, 6],
        [3, 6, 2],
        [7, 9, 6],

        # Dachflaeche links
        [4, 8, 9],
        [4, 9, 7],

        # Dachflaeche rechts
        [5, 6, 9],
        [5, 9, 8],
    ])

    # ------------------------------------------------------------------
    # STL-Mesh erzeugen
    # ------------------------------------------------------------------
    building = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            building.vectors[i][j] = vertices[f[j]]

    # ------------------------------------------------------------------
    # Rotation um Z-Achse (Windrichtung)
    # ------------------------------------------------------------------
    if abs(angle_deg) > 1e-6:
        building.rotate([0.0, 0.0, 1.0], math.radians(angle_deg))

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    building.save(filepath)

    # ------------------------------------------------------------------
    # Bounding Box berechnen und zurueckgeben
    # ------------------------------------------------------------------
    bbox = (
        float(building.x.min()), float(building.x.max()),
        float(building.y.min()), float(building.y.max()),
        float(building.z.min()), float(building.z.max()),
    )

    return bbox


# ======================================================================
# Standalone-Test
# ======================================================================
if __name__ == "__main__":
    test_path = "test_building.stl"
    bbox = create_building(10.0, 10.0, 6.0, 2.0, 45.0, test_path)
    print(f"STL gespeichert: {test_path}")
    print(f"Bounding Box:    {bbox}")
    print(f"H (gesamt):      {6.0 + 2.0} m")
