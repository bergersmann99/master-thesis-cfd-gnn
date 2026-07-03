"""
trainGATv2_efficient.py
=======================
Trainiert ein Graph Attention Network v2 (GATv2) als Surrogat-Modell
fuer stationaere RANS-Stroemungsfelder um ein parametrisches
Satteldach-Gebaeude.

VRAM-optimierte Variante von trainGATv2.py mit zwei zusaetzlichen
Optionen:

  1. Gradient Checkpointing (--gradient-checkpointing):
     Aktivierungen im Processor werden waehrend des Forward-Passes
     nicht gespeichert, sondern beim Backward-Pass neu berechnet.
     Reduziert den VRAM-Bedarf von O(L * E * H) auf O(E * H),
     wobei L die Anzahl der Schichten ist.
     Referenz: Chen et al. (2016), "Training Deep Nets with Sublinear
     Memory Cost".

  2. Mixed Precision Training mit BF16 (--mixed-precision):
     Forward-Pass und Loss-Berechnung in bfloat16, Gradienten und
     Optimizer-State in float32. Halbiert den Speicherbedarf der
     Aktivierungen bei minimalem Genauigkeitsverlust.
     BF16 statt FP16, weil die RTX PRO 6000 Blackwell BF16 nativ
     unterstuetzt und BF16 keinen Loss Scaling benoetigt.
     Referenz: Micikevicius et al. (2018), "Mixed Precision Training".

Architektur: Encode-Process-Decode (Pfaff et al., 2020)
    - Encoder:   MLP (13 -> hidden_dim)
    - Processor: M Runden GATv2Conv mit Residual + LayerNorm
    - Decoder:   MLP (hidden_dim -> 6)

VARIANTE OHNE cell_volume:
    Trainiert auf graph-dataset_medium_no_cellvol — Input-Features
    auf 13 reduziert (cell_volume entfernt).

Input-Features (13):  x, y, z, wall_distance,
                      node_type (7x One-Hot), U_ref, angle
Output-Features (6):  Ux, Uy, Uz, p, k, epsilon

Normalisierung: z-Score (berechnet auf Trainingsdaten)
Loss:           MSE auf normalisierten Targets
Optimizer:      Adam (Kingma und Ba, 2015)
Scheduler:      ReduceLROnPlateau

Verwendung:
    python trainGATv2_efficient.py --data-dir ./graph_dataset
    python trainGATv2_efficient.py --data-dir ./graph_dataset \\
        --gradient-checkpointing --mixed-precision
    python trainGATv2_efficient.py --s3-download \\
        --s3-prefix graph-dataset_building_focus_25 \\
        --gradient-checkpointing --mixed-precision

Siehe QUELLEN_trainGATv2_efficient.md fuer vollstaendige Quellenangaben.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


# ======================================================================
# Reproduzierbarkeit
# ======================================================================

def set_seed(seed):
    """
    Setzt alle Random Seeds fuer vollstaendige Reproduzierbarkeit.

    Parameter
    ---------
    seed : int
        Globaler Seed-Wert.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Logging-Konfiguration
# ======================================================================

def setup_file_logger(output_dir):
    """
    Konfiguriert einen reinen Datei-Logger.

    Alle Trainingsmetriken, Hyperparameter und Ergebnisse werden
    lueckenlos in einer Log-Datei protokolliert, damit keine
    Information durch Context Compaction verloren geht.

    Die Konsolenausgabe wird separat ueber sys.stdout gesteuert,
    um live-ueberschreibende Fortschrittszeilen zu ermoeglichen.

    Parameter
    ---------
    output_dir : str
        Verzeichnis fuer Log-Dateien und Checkpoints.

    Rueckgabe
    ---------
    logging.Logger
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    logger = logging.getLogger("trainGATv2_efficient")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


def log_and_print(logger, msg):
    """
    Schreibt eine Nachricht in die Log-Datei UND gibt sie
    als vollstaendige Zeile in der Konsole aus.

    Vor der Ausgabe wird die aktuelle Konsolenzeile geloescht,
    damit live-Fortschrittsanzeigen sauber ueberschrieben werden.

    Parameter
    ---------
    logger : logging.Logger
        Datei-Logger.
    msg : str
        Nachricht.
    """
    logger.info(msg)
    sys.stdout.write(f"\r{' ' * 80}\r")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def log_only(logger, msg):
    """
    Schreibt eine Nachricht NUR in die Log-Datei (nicht Konsole).

    Parameter
    ---------
    logger : logging.Logger
        Datei-Logger.
    msg : str
        Nachricht.
    """
    logger.info(msg)


def console_live(msg):
    """
    Ueberschreibt die aktuelle Konsolenzeile mit einer
    Live-Fortschrittsanzeige (ohne Zeilenumbruch).

    Parameter
    ---------
    msg : str
        Fortschrittsnachricht.
    """
    sys.stdout.write(f"\r{msg:<80}")
    sys.stdout.flush()


# ======================================================================
# S3-Hilfsfunktionen
# ======================================================================

def download_from_s3(s3_path, local_path, max_retries=3):
    """
    Laedt eine Datei von S3 herunter mit Retry-Logik.

    Parameter
    ---------
    s3_path : str
        S3-URI (z.B. s3://bucket/prefix/train.pt).
    local_path : str
        Lokaler Zielpfad.
    max_retries : int
        Maximale Download-Versuche.

    Rueckgabe
    ---------
    bool : True bei Erfolg, False bei Fehler.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", s3_path, local_path,
                 "--cli-read-timeout", "120",
                 "--cli-connect-timeout", "30"],
                check=True,
                capture_output=True,
            )
            if os.path.exists(local_path):
                return True
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"  WARNUNG: Download fehlgeschlagen "
                      f"(Versuch {attempt}/{max_retries}), "
                      f"Retry in {wait}s...")
                time.sleep(wait)
            else:
                err_msg = str(e)
                if hasattr(e, "stderr") and e.stderr:
                    err_msg = e.stderr.decode("utf-8", errors="replace")
                print(f"  FEHLER: Download endgueltig fehlgeschlagen: "
                      f"{err_msg}")
    return False


def download_dataset_from_s3(s3_bucket, s3_prefix, local_dir):
    """
    Laedt train.pt, val.pt, test.pt und metadata.yaml von S3.

    Parameter
    ---------
    s3_bucket : str
        S3-Bucket-Name.
    s3_prefix : str
        Pfad-Prefix im Bucket (z.B. graph-dataset).
    local_dir : str
        Lokales Zielverzeichnis.
    """
    files = ["train.pt", "val.pt", "test.pt", "metadata.yaml"]
    s3_base = f"s3://{s3_bucket}/{s3_prefix}"

    print(f"\n  Lade Graph-Datensatz von {s3_base}/...")
    os.makedirs(local_dir, exist_ok=True)

    for fname in files:
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path):
            print(f"  [SKIP] {fname} existiert bereits lokal.")
            continue

        s3_path = f"{s3_base}/{fname}"
        print(f"  Lade {fname}...", end=" ", flush=True)
        ok = download_from_s3(s3_path, local_path)
        if ok:
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        else:
            print("FEHLER")
            sys.exit(1)

    print(f"  Download abgeschlossen.\n")


# ======================================================================
# Datensatz laden und Normalisierung
# ======================================================================

def load_dataset(data_dir):
    """
    Laedt die Graph-Datensatz-Splits von der Festplatte.

    Parameter
    ---------
    data_dir : str
        Verzeichnis mit train.pt, val.pt, test.pt.

    Rueckgabe
    ---------
    tuple : (train_list, val_list, test_list)
        Listen von torch_geometric.data.Data Objekten.
    """
    train = torch.load(os.path.join(data_dir, "train.pt"),
                       weights_only=False)
    val = torch.load(os.path.join(data_dir, "val.pt"),
                     weights_only=False)
    test = torch.load(os.path.join(data_dir, "test.pt"),
                      weights_only=False)
    return train, val, test


def compute_normalization_stats(data_list):
    """
    Berechnet Mittelwert und Standardabweichung pro Feature
    ueber alle Knoten aller Graphen im Datensatz.

    Parameter
    ---------
    data_list : list[Data]
        Liste von Graph-Objekten (Trainingsdaten).

    Rueckgabe
    ---------
    dict : {
        'x_mean': Tensor (14,), 'x_std': Tensor (14,),
        'y_mean': Tensor (6,),  'y_std': Tensor (6,)
    }
    """
    all_x = torch.cat([d.x for d in data_list], dim=0)
    all_y = torch.cat([d.y for d in data_list], dim=0)

    x_mean = all_x.mean(dim=0)
    x_std = all_x.std(dim=0)
    y_mean = all_y.mean(dim=0)
    y_std = all_y.std(dim=0)

    # Nulldivision verhindern (z.B. bei One-Hot-Features mit
    # konstanten Spalten)
    eps = 1e-8
    x_std = torch.clamp(x_std, min=eps)
    y_std = torch.clamp(y_std, min=eps)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_dataset(data_list, stats):
    """
    Wendet z-Score-Normalisierung auf Input- und Output-Features an.

    x_norm = (x - mean) / std

    Die Normalisierung erfolgt in-place auf den Tensoren.

    Parameter
    ---------
    data_list : list[Data]
        Liste von Graph-Objekten.
    stats : dict
        Normalisierungsstatistiken aus compute_normalization_stats().

    Rueckgabe
    ---------
    list[Data] : Die normalisierten Graph-Objekte (in-place).
    """
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    for data in data_list:
        data.x = (data.x - x_mean) / x_std
        data.y = (data.y - y_mean) / y_std

    return data_list


# ======================================================================
# Modell-Definition: Encode-Process-Decode mit GATv2Conv
# ======================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron mit konfigurierbarer Tiefe.

    Aufbau: Linear -> ReLU -> Linear -> ReLU -> Linear
    (2 versteckte Schichten, letzte Schicht ohne Aktivierung)

    Referenz: Standard-MLP-Architektur, verwendet in
    MeshGraphNets (Pfaff et al., 2020) fuer Encoder/Decoder.
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        """
        Parameter
        ---------
        in_dim : int
            Eingabedimension.
        hidden_dim : int
            Dimension der versteckten Schichten.
        out_dim : int
            Ausgabedimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GATv2Processor(nn.Module):
    """
    Processor-Block: Mehrere Runden GATv2 Message Passing
    mit Multi-Head Attention, Residual Connections und
    Layer Normalization.

    Optional mit Gradient Checkpointing: Aktivierungen werden
    im Forward-Pass nicht gespeichert, sondern beim Backward-Pass
    neu berechnet. Dies reduziert den VRAM-Bedarf von O(L * E * H)
    auf O(E * H) auf Kosten von ~30% laengerem Training.

    Jede Runde:
        h' = LayerNorm(ReLU(GATv2Conv(h)) + h)

    Multi-Head Attention:
        Jede GATv2Conv-Schicht nutzt mehrere Attention-Koepfe.
        Die Dimension pro Kopf ist hidden_dim // heads, sodass
        die konkatenierte Ausgabe hidden_dim ergibt. Dies
        ermoeglicht stabile Residual Connections bei gleichzeitiger
        Multi-Head Expressivitaet.

    GATv2 Attention-Koeffizienten (Brody et al., 2022):
        e_ij = a^T * LeakyReLU(W * [h_i || h_j])
        alpha_ij = softmax_j(e_ij)

    Referenzen:
        - Chen et al. (2016): Gradient Checkpointing
        - Brody et al. (2022): GATv2, dynamische Attention
        - Velickovic et al. (2018): Originales GAT
        - Pfaff et al. (2020): Residual-Struktur im Processor
        - Ba et al. (2016): Layer Normalization
    """

    def __init__(self, hidden_dim, num_layers, heads=4, dropout=0.0,
                 attention_dropout=0.0, use_gradient_checkpointing=False):
        """
        Parameter
        ---------
        hidden_dim : int
            Dimension des latenten Raums.
        num_layers : int
            Anzahl Message-Passing-Runden.
        heads : int
            Anzahl Attention-Koepfe pro Schicht.
            hidden_dim muss durch heads teilbar sein.
        dropout : float
            Dropout-Rate auf Features (0.0 = kein Dropout).
        attention_dropout : float
            Dropout-Rate auf Attention-Koeffizienten.
        use_gradient_checkpointing : bool
            Wenn True, werden Aktivierungen nicht gespeichert
            sondern beim Backward-Pass neu berechnet.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing

        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) muss durch heads ({heads}) "
            f"teilbar sein.")
        dim_per_head = hidden_dim // heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=dim_per_head,
                heads=heads,
                concat=True,
                dropout=attention_dropout,
                add_self_loops=True,
                share_weights=False,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def _run_layer(self, x, edge_index, layer_idx):
        """
        Fuehrt eine einzelne Message-Passing-Runde durch.

        Diese Methode wird entweder direkt oder ueber
        torch.utils.checkpoint.checkpoint aufgerufen.

        Parameter
        ---------
        x : Tensor (N, hidden_dim)
            Knoten-Features.
        edge_index : Tensor (2, E)
            Kanten-Indizes.
        layer_idx : int
            Index der aktuellen Schicht.

        Rueckgabe
        ---------
        Tensor (N, hidden_dim) : Verarbeitete Knoten-Features.
        """
        conv = self.convs[layer_idx]
        norm = self.norms[layer_idx]

        x_residual = x
        x = conv(x, edge_index)
        x = F.relu(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
        x = norm(x)
        x = x + x_residual

        return x

    def forward(self, x, edge_index):
        """
        Parameter
        ---------
        x : Tensor (N, hidden_dim)
            Knoten-Features im latenten Raum.
        edge_index : Tensor (2, E)
            Kanten-Indizes.

        Rueckgabe
        ---------
        Tensor (N, hidden_dim) : Verarbeitete Knoten-Features.
        """
        for i in range(self.num_layers):
            if self.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    partial(self._run_layer, layer_idx=i),
                    x, edge_index,
                    use_reentrant=False,
                )
            else:
                x = self._run_layer(x, edge_index, i)

        return x


class GATv2Surrogate(nn.Module):
    """
    Vollstaendiges Encode-Process-Decode Surrogat-Modell mit GATv2.

    Architektur:
        1. Encoder: MLP projiziert 14 Input-Features in den
           latenten Raum (hidden_dim).
        2. Processor: M Runden GATv2Conv mit Multi-Head Attention,
           Residual Connections und LayerNorm fuehren Message
           Passing im latenten Raum durch.
        3. Decoder: MLP projiziert latente Knoten-Features
           zurueck auf 6 physikalische Ausgangsgroessen.

    Referenzen:
        - Pfaff et al. (2020): Encode-Process-Decode Paradigma
        - Brody et al. (2022): GATv2, dynamische Attention
        - Velickovic et al. (2018): Graph Attention Networks
        - Gladstone et al. (2024): GNN-Surrogat fuer stationaere PDEs
    """

    def __init__(self, in_dim=14, out_dim=6, hidden_dim=128,
                 num_layers=10, heads=4, dropout=0.0,
                 attention_dropout=0.0,
                 use_gradient_checkpointing=False):
        """
        Parameter
        ---------
        in_dim : int
            Anzahl Input-Features pro Knoten (Standard: 14).
        out_dim : int
            Anzahl Output-Features pro Knoten (Standard: 6).
        hidden_dim : int
            Dimension des latenten Raums.
        num_layers : int
            Anzahl Message-Passing-Runden im Processor.
        heads : int
            Anzahl Attention-Koepfe pro GATv2-Schicht.
        dropout : float
            Dropout-Rate auf Features im Processor.
        attention_dropout : float
            Dropout-Rate auf Attention-Koeffizienten.
        use_gradient_checkpointing : bool
            Gradient Checkpointing im Processor aktivieren.
        """
        super().__init__()

        self.encoder = MLP(in_dim, hidden_dim, hidden_dim)
        self.processor = GATv2Processor(
            hidden_dim, num_layers, heads, dropout, attention_dropout,
            use_gradient_checkpointing=use_gradient_checkpointing)
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index):
        """
        Parameter
        ---------
        x : Tensor (N, 14)
            Normalisierte Input-Features.
        edge_index : Tensor (2, E)
            Kanten-Indizes (bidirektional, k-NN).

        Rueckgabe
        ---------
        Tensor (N, 6) : Vorhergesagte (normalisierte) Output-Features.
        """
        # Encode: Input-Features -> latenter Raum
        h = self.encoder(x)

        # Process: Message Passing mit Attention im latenten Raum
        h = self.processor(h, edge_index)

        # Decode: Latenter Raum -> physikalische Groessen
        out = self.decoder(h)

        return out


# ======================================================================
# Training
# ======================================================================

def train_one_epoch(model, loader, optimizer, device,
                    use_mixed_precision=False):
    """
    Fuehrt eine Trainings-Epoche durch.

    Parameter
    ---------
    model : GATv2Surrogate
        Das Modell.
    loader : DataLoader
        PyTorch Geometric DataLoader mit Trainingsgraphen.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Rechengeraet (CPU/GPU).
    use_mixed_precision : bool
        BF16 Mixed Precision aktivieren.

    Rueckgabe
    ---------
    float : Durchschnittlicher Loss ueber alle Batches.
    """
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_mixed_precision and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(batch.x, batch.edge_index)
                loss = F.mse_loss(pred, batch.y, reduction="sum")
        else:
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y, reduction="sum")

        loss.backward()

        # Gradient Clipping gegen explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        n_nodes = batch.x.size(0)
        total_loss += loss.item()
        total_nodes += n_nodes

    avg_loss = total_loss / max(total_nodes, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, use_mixed_precision=False):
    """
    Evaluiert das Modell auf einem Datensatz.

    Parameter
    ---------
    model : GATv2Surrogate
        Das Modell.
    loader : DataLoader
        DataLoader mit Evaluierungsgraphen.
    device : torch.device
        Rechengeraet.
    use_mixed_precision : bool
        BF16 Mixed Precision aktivieren.

    Rueckgabe
    ---------
    float : Durchschnittlicher MSE-Loss (auf normalisierten Daten).
    """
    model.eval()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)

        if use_mixed_precision and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(batch.x, batch.edge_index)
                loss = F.mse_loss(pred, batch.y, reduction="sum")
        else:
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y, reduction="sum")

        total_loss += loss.item()
        total_nodes += batch.x.size(0)

    avg_loss = total_loss / max(total_nodes, 1)
    return avg_loss


# ======================================================================
# Detaillierte Evaluation (denormalisiert)
# ======================================================================

@torch.no_grad()
def evaluate_detailed(model, data_list, stats, device):
    """
    Berechnet detaillierte Metriken auf denormalisierten Vorhersagen.

    Metriken pro Feld:
        - MSE:  Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAE:  Mean Absolute Error
        - R2:   Bestimmtheitsmass
        - rL2:  Relative L2-Norm (||pred - true||_2 / ||true||_2)

    Parameter
    ---------
    model : GATv2Surrogate
        Das trainierte Modell.
    data_list : list[Data]
        Liste der (normalisierten) Graph-Objekte.
    stats : dict
        Normalisierungsstatistiken (fuer Denormalisierung).
    device : torch.device
        Rechengeraet.

    Rueckgabe
    ---------
    dict : Metriken pro Feld und Gesamt.
    """
    model.eval()
    field_names = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
    y_mean = stats["y_mean"].to(device)
    y_std = stats["y_std"].to(device)

    all_pred = []
    all_true = []

    for data in data_list:
        data = data.to(device)
        pred_norm = model(data.x, data.edge_index)

        # Denormalisieren
        pred = pred_norm * y_std + y_mean
        true = data.y * y_std + y_mean

        all_pred.append(pred.cpu())
        all_true.append(true.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    metrics = {}

    for i, name in enumerate(field_names):
        p = all_pred[:, i]
        t = all_true[:, i]

        mse = F.mse_loss(p, t).item()
        rmse = mse ** 0.5
        mae = (p - t).abs().mean().item()

        # R2 (Bestimmtheitsmass)
        ss_res = ((t - p) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        # Relative L2-Norm
        rl2 = (((p - t) ** 2).sum().sqrt()
               / max(((t) ** 2).sum().sqrt(), 1e-12)).item()

        metrics[name] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "rL2": rl2,
        }

    # Gesamtmetriken (gemittelt ueber alle Felder)
    metrics["gesamt"] = {
        "MSE": np.mean([metrics[n]["MSE"] for n in field_names]),
        "RMSE": np.mean([metrics[n]["RMSE"] for n in field_names]),
        "MAE": np.mean([metrics[n]["MAE"] for n in field_names]),
        "R2": np.mean([metrics[n]["R2"] for n in field_names]),
        "rL2": np.mean([metrics[n]["rL2"] for n in field_names]),
    }

    return metrics


# ======================================================================
# Checkpoint Speichern / Laden
# ======================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                    stats, args, output_dir, filename="best_model.pt"):
    """
    Speichert einen vollstaendigen Modell-Checkpoint.

    Enthaelt Modellgewichte, Optimizer-State, Scheduler-State,
    Normalisierungsstatistiken und Hyperparameter — alles was
    fuer die Wiederaufnahme des Trainings oder Inferenz noetig ist.

    Parameter
    ---------
    model : GATv2Surrogate
        Das Modell.
    optimizer : Optimizer
        Adam Optimizer.
    scheduler : ReduceLROnPlateau
        LR-Scheduler.
    epoch : int
        Aktuelle Epoche.
    val_loss : float
        Bester Validierungs-Loss.
    stats : dict
        Normalisierungsstatistiken.
    args : Namespace
        CLI-Argumente (Hyperparameter).
    output_dir : str
        Speicherverzeichnis.
    filename : str
        Dateiname des Checkpoints.
    """
    path = os.path.join(output_dir, filename)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "norm_stats": {
            k: v.cpu() for k, v in stats.items()
        },
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "attention_dropout": args.attention_dropout,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
            "mixed_precision": args.mixed_precision,
        },
        "architecture": "GATv2_Encode-Process-Decode_efficient",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, path)


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # CLI-Argumente
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="GATv2 Surrogat-Modell Training "
                    "(Encode-Process-Decode, VRAM-optimiert)"
    )

    # Daten
    parser.add_argument(
        "--data-dir", type=str, default="./graph_dataset",
        help="Lokales Verzeichnis mit train.pt, val.pt, test.pt "
             "(Standard: ./graph_dataset)")
    parser.add_argument(
        "--s3-download", action="store_true",
        help="Graph-Datensatz von S3 herunterladen")
    parser.add_argument(
        "--s3-bucket", type=str, default="amzn-master-sim-bucket",
        help="S3-Bucket (Standard: amzn-master-sim-bucket)")
    parser.add_argument(
        "--s3-prefix", type=str, default="graph-dataset",
        help="S3-Prefix (Standard: graph-dataset)")

    # Modell-Architektur
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Dimension des latenten Raums (Standard: 128)")
    parser.add_argument(
        "--num-layers", type=int, default=10,
        help="Anzahl GATv2 Message-Passing-Runden (Standard: 10)")
    parser.add_argument(
        "--heads", type=int, default=4,
        help="Anzahl Attention-Koepfe pro Schicht (Standard: 4). "
             "hidden-dim muss durch heads teilbar sein.")
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout-Rate auf Features (Standard: 0.0)")
    parser.add_argument(
        "--attention-dropout", type=float, default=0.0,
        help="Dropout-Rate auf Attention-Koeffizienten "
             "(Standard: 0.0)")

    # Training
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Maximale Anzahl Epochen (Standard: 500)")
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initiale Lernrate (Standard: 1e-4)")
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch-Groesse in Graphen (Standard: 4)")
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early-Stopping Geduld in Epochen (Standard: 50)")
    parser.add_argument(
        "--min-lr", type=float, default=1e-6,
        help="Minimale Lernrate fuer Scheduler (Standard: 1e-6)")

    # VRAM-Optimierung (NEU)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Gradient Checkpointing aktivieren: Aktivierungen "
             "werden nicht gespeichert, sondern beim Backward-Pass "
             "neu berechnet. Reduziert VRAM um ~60-80%% im Processor "
             "auf Kosten von ~30%% laengerem Training.")
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Mixed Precision Training (BF16) aktivieren: "
             "Forward-Pass in bfloat16, Gradienten in float32. "
             "Halbiert den Speicherbedarf der Aktivierungen.")

    # Allgemein
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random Seed (Standard: 42)")
    parser.add_argument(
        "--output-dir", type=str, default="./output_gatv2_efficient",
        help="Ausgabeverzeichnis "
             "(Standard: ./output_gatv2_efficient)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validierung: hidden_dim muss durch heads teilbar sein
    # ------------------------------------------------------------------
    if args.hidden_dim % args.heads != 0:
        print(f"FEHLER: hidden-dim ({args.hidden_dim}) muss durch "
              f"heads ({args.heads}) teilbar sein.")
        print(f"  Vorschlaege: --heads 4 (32 dim/Kopf) oder "
              f"--heads 8 (16 dim/Kopf)")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)
    logger = setup_file_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_mixed_precision = (args.mixed_precision
                           and device.type == "cuda")
    use_grad_ckpt = args.gradient_checkpointing

    dim_per_head = args.hidden_dim // args.heads

    # Header (Log + Konsole)
    header_lines = [
        "",
        "=" * 60,
        "   GATv2 SURROGAT-MODELL TRAINING (VRAM-optimiert)",
        "   Encode-Process-Decode (Pfaff et al., 2020)",
        "   Processor: GATv2Conv (Brody et al., 2022)",
        "=" * 60,
        "",
        f"   Geraet:            {device}",
    ]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        header_lines.append(f"   GPU:               {gpu_name}")
        header_lines.append(f"   VRAM:              {vram_gb:.1f} GB")

    header_lines += [
        f"   Random Seed:       {args.seed}",
        "",
        "   --- Hyperparameter ---",
        f"   Hidden Dim:        {args.hidden_dim}",
        f"   MP-Runden:         {args.num_layers}",
        f"   Attention-Koepfe:  {args.heads} "
        f"({dim_per_head} dim/Kopf)",
        f"   Dropout:           {args.dropout}",
        f"   Attn-Dropout:      {args.attention_dropout}",
        f"   Lernrate:          {args.lr}",
        f"   Min. Lernrate:     {args.min_lr}",
        f"   Batch-Groesse:     {args.batch_size}",
        f"   Max. Epochen:      {args.epochs}",
        f"   Early-Stop:        {args.patience} Epochen",
        f"   Ausgabe:           {args.output_dir}",
        "",
        "   --- VRAM-Optimierung ---",
        f"   Gradient Checkpointing: "
        f"{'AKTIV' if use_grad_ckpt else 'deaktiviert'}",
        f"   Mixed Precision (BF16): "
        f"{'AKTIV' if use_mixed_precision else 'deaktiviert'}",
        "",
    ]

    for line in header_lines:
        log_and_print(logger, line)

    # ------------------------------------------------------------------
    # Daten laden
    # ------------------------------------------------------------------
    if args.s3_download:
        download_dataset_from_s3(
            args.s3_bucket, args.s3_prefix, args.data_dir)

    log_and_print(logger, "   Lade Datensatz...")
    train_data, val_data, test_data = load_dataset(args.data_dir)
    log_and_print(logger, f"   Train:  {len(train_data)} Graphen")
    log_and_print(logger, f"   Val:    {len(val_data)} Graphen")
    log_and_print(logger, f"   Test:   {len(test_data)} Graphen")

    # Datensatz-Statistiken
    train_nodes = sum(d.x.size(0) for d in train_data)
    val_nodes = sum(d.x.size(0) for d in val_data)
    test_nodes = sum(d.x.size(0) for d in test_data)
    train_edges = sum(d.edge_index.size(1) for d in train_data)
    log_and_print(logger, f"   Train Knoten: {train_nodes:,}")
    log_and_print(logger, f"   Val Knoten:   {val_nodes:,}")
    log_and_print(logger, f"   Test Knoten:  {test_nodes:,}")
    log_and_print(logger, f"   Train Kanten: {train_edges:,}")
    log_and_print(logger, "")

    # ------------------------------------------------------------------
    # Checkpoint pruefen (vor Normalisierung, da Stats darin gespeichert)
    # ------------------------------------------------------------------
    resume_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    resume_checkpoint = None
    if os.path.exists(resume_checkpoint_path):
        resume_checkpoint = torch.load(
            resume_checkpoint_path, weights_only=False)
        log_and_print(logger, "   Vorhandener Checkpoint gefunden — "
                              "Resume-Modus aktiv.")

    # ------------------------------------------------------------------
    # Normalisierung
    # ------------------------------------------------------------------
    if resume_checkpoint is not None:
        stats = {k: v for k, v in resume_checkpoint["norm_stats"].items()}
        log_and_print(logger,
                      "   Normalisierungsstatistiken aus Checkpoint geladen.")
    else:
        log_and_print(logger,
                      "   Berechne Normalisierungsstatistiken (Train)...")
        stats = compute_normalization_stats(train_data)
        log_and_print(logger,
                      f"   x_mean: {stats['x_mean'].numpy()}")
        log_and_print(logger,
                      f"   x_std:  {stats['x_std'].numpy()}")
        log_and_print(logger,
                      f"   y_mean: {stats['y_mean'].numpy()}")
        log_and_print(logger,
                      f"   y_std:  {stats['y_std'].numpy()}")
    log_and_print(logger, "")

    train_data = normalize_dataset(train_data, stats)
    val_data = normalize_dataset(val_data, stats)
    test_data = normalize_dataset(test_data, stats)
    log_and_print(logger,
                  "   Normalisierung angewendet "
                  "(z-Score, Train-basiert).")
    log_and_print(logger, "")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Modell, Optimizer, Scheduler
    # ------------------------------------------------------------------
    model = GATv2Surrogate(
        in_dim=13,
        out_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        use_gradient_checkpointing=use_grad_ckpt,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    log_and_print(logger, f"   Modell erstellt: GATv2Surrogate")
    log_and_print(logger,
                  f"   Parameter:       {n_params:,} "
                  f"({n_trainable:,} trainierbar)")
    log_and_print(logger, "")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=args.min_lr,
    )

    # ------------------------------------------------------------------
    # Resume-Zustand wiederherstellen
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_history = []
    val_history = []

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_loss = resume_checkpoint["val_loss"]

        history_path = os.path.join(args.output_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                hist = json.load(f)
            train_history = hist["train_loss"][:resume_checkpoint["epoch"]]
            val_history = hist["val_loss"][:resume_checkpoint["epoch"]]

        log_and_print(logger,
                      f"   Fortsetze ab Epoche {start_epoch} "
                      f"(bester Val-Loss: {best_val_loss:.6f}).")
        log_and_print(logger, "")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    log_and_print(logger, "=" * 60)
    log_and_print(logger, "   STARTE TRAINING")
    log_and_print(logger, "=" * 60)
    log_and_print(logger, "")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Live-Fortschritt: Training laeuft
        console_live(
            f"   Epoche {epoch}/{args.epochs}: Training...")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            use_mixed_precision=use_mixed_precision)

        # Live-Fortschritt: Validierung laeuft
        console_live(
            f"   Epoche {epoch}/{args.epochs}: Validierung...")

        val_loss = evaluate(
            model, val_loader, device,
            use_mixed_precision=use_mixed_precision)

        # Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Historien speichern
        train_history.append(train_loss)
        val_history.append(val_loss)

        # NaN-Erkennung
        if np.isnan(train_loss) or np.isnan(val_loss):
            log_and_print(
                logger,
                f"   ABBRUCH: NaN im Loss (Epoche {epoch}) | "
                f"Train: {train_loss} | Val: {val_loss}")
            break

        # Bestes Modell pruefen
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, stats, args, args.output_dir)
        else:
            epochs_without_improvement += 1

        # ======================================================
        # Logging-Strategie:
        #   Log-Datei:  JEDE Epoche (lueckenlos)
        #   Konsole:    Live-Zeile wird ueberschrieben,
        #               permanente Zeile nur bei Verbesserung
        #               oder LR-Wechsel
        # ======================================================

        log_msg = (
            f"   Epoche {epoch:4d}/{args.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
            f"{' *' if improved else ''}")

        # Log-Datei: jede Epoche
        log_only(logger, log_msg)

        # Konsole: Live-Zeile (wird naechste Epoche ueberschrieben)
        elapsed = time.time() - training_start
        elapsed_str = (f"{int(elapsed // 60)}m "
                       f"{int(elapsed % 60)}s")
        console_live(
            f"   [{epoch}/{args.epochs}] "
            f"T:{train_loss:.5f} V:{val_loss:.5f} "
            f"best:{best_val_loss:.5f} "
            f"LR:{current_lr:.1e} "
            f"| {elapsed_str}")

        # Konsole: Permanente Zeile bei Verbesserung
        if improved:
            sys.stdout.write(
                f"\r   [*] Epoche {epoch:4d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} (neues Minimum) | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
                f"{' ' * 10}\n")
            sys.stdout.flush()

        # Konsole: LR-Wechsel melden
        if new_lr < current_lr:
            lr_msg = (f"   >> Lernrate reduziert: "
                      f"{current_lr:.2e} -> {new_lr:.2e}")
            log_only(logger, lr_msg)
            sys.stdout.write(f"\r{lr_msg}{' ' * 30}\n")
            sys.stdout.flush()

        # Early Stopping
        if epochs_without_improvement >= args.patience:
            es_msg = (
                f"\n   Early Stopping nach {epoch} Epochen "
                f"(keine Verbesserung seit "
                f"{args.patience} Epochen)")
            log_and_print(logger, es_msg)
            break

    # Letzte Live-Zeile loeschen
    sys.stdout.write(f"\r{' ' * 80}\r")
    sys.stdout.flush()

    training_time = time.time() - training_start
    peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    log_and_print(logger, f"   Peak VRAM:       {peak_vram_bytes/(1024**3):.2f} GB ({peak_vram_bytes} Bytes)")
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)

    log_and_print(logger, "")
    log_and_print(logger,
                  f"   Training abgeschlossen in "
                  f"{minutes}min {seconds}s")
    log_and_print(logger,
                  f"   Bester Val-Loss: {best_val_loss:.6f}")
    log_and_print(logger, "")

    # ------------------------------------------------------------------
    # Trainingshistorie speichern
    # ------------------------------------------------------------------
    history_path = os.path.join(args.output_dir,
                                "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "train_loss": train_history,
            "val_loss": val_history,
        }, f, indent=2)
    log_and_print(logger,
                  f"   Trainingshistorie gespeichert: {history_path}")

    # ------------------------------------------------------------------
    # Bestes Modell laden und evaluieren
    # ------------------------------------------------------------------
    log_and_print(logger, "")
    log_and_print(logger, "=" * 60)
    log_and_print(logger, "   EVALUATION (bestes Modell)")
    log_and_print(logger, "=" * 60)
    log_and_print(logger, "")

    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        log_and_print(
            logger,
            f"   Bestes Modell geladen "
            f"(Epoche {checkpoint['epoch']})")
    else:
        log_and_print(
            logger,
            "   WARNUNG: Kein Checkpoint gefunden, "
            "verwende aktuelles Modell.")

    # Evaluation auf allen Splits
    for split_name, split_data in [("Train", train_data),
                                   ("Val", val_data),
                                   ("Test", test_data)]:
        metrics = evaluate_detailed(
            model, split_data, stats, device)

        log_and_print(logger, f"\n   --- {split_name} ---")
        header = (f"   {'Feld':<10s} {'MSE':>12s} "
                  f"{'RMSE':>12s} {'MAE':>12s} "
                  f"{'R²':>10s} {'rL2':>10s}")
        log_and_print(logger, header)
        log_and_print(logger, f"   {'-' * 66}")

        field_names = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
        for name in field_names:
            m = metrics[name]
            row = (f"   {name:<10s} {m['MSE']:>12.6f} "
                   f"{m['RMSE']:>12.6f} {m['MAE']:>12.6f} "
                   f"{m['R2']:>10.6f} {m['rL2']:>10.6f}")
            log_and_print(logger, row)

        m = metrics["gesamt"]
        log_and_print(logger, f"   {'-' * 66}")
        total_row = (f"   {'GESAMT':<10s} {m['MSE']:>12.6f} "
                     f"{m['RMSE']:>12.6f} {m['MAE']:>12.6f} "
                     f"{m['R2']:>10.6f} {m['rL2']:>10.6f}")
        log_and_print(logger, total_row)

    # Metriken als JSON speichern
    test_metrics = evaluate_detailed(
        model, test_data, stats, device)
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    log_and_print(logger,
                  f"\n   Test-Metriken gespeichert: {metrics_path}")

    # ------------------------------------------------------------------
    # Zusammenfassung
    # ------------------------------------------------------------------
    summary_lines = [
        "",
        "=" * 60,
        "   ZUSAMMENFASSUNG",
        "=" * 60,
        "",
        f"   Architektur:       GATv2 Encode-Process-Decode "
        f"(efficient)",
        f"   Parameter:         {n_params:,}",
        f"   Hidden Dim:        {args.hidden_dim}",
        f"   MP-Runden:         {args.num_layers}",
        f"   Attention-Koepfe:  {args.heads} "
        f"({dim_per_head} dim/Kopf)",
        f"   Grad. Checkpoint:  "
        f"{'AKTIV' if use_grad_ckpt else 'deaktiviert'}",
        f"   Mixed Precision:   "
        f"{'BF16' if use_mixed_precision else 'deaktiviert'}",
        f"   Trainingszeit:     {minutes}min {seconds}s",
        f"   Bester Val-Loss:   {best_val_loss:.6f}",
        f"   Test R² (gesamt):  "
        f"{test_metrics['gesamt']['R2']:.6f}",
        f"   Test rL2 (gesamt): "
        f"{test_metrics['gesamt']['rL2']:.6f}",
        f"   Ausgabe:           {args.output_dir}",
        "",
        "=" * 60,
        "   FERTIG",
        "=" * 60,
    ]

    for line in summary_lines:
        log_and_print(logger, line)


if __name__ == "__main__":
    main()
