"""
BulkCell-GNN — Bipartite Graph Neural Network for Bulk-to-Single-Cell Expression Transfer
==========================================================================================
Course project: COMP/EECE 7/8740 Neural Networks

Datasets:
  Bulk:  GSE39582  — 536 CRC samples, MSI/MSS labels, ~22,880 genes
  Cell:  GSE132465 — 63,689 CRC single cells, 23 patients, ~20k genes

Graph structure:
  B-B edges: bulk sample similarity (cosine on GSVA/expression)
  C-C edges: cell kNN (standard scRNA-seq preprocessing)
  B-C edges: cross-modality alignment (cosine on shared genes, top-K sparse)

Novel contributions:
  1. Cell-type-aware C→B pooling with learnable type-attention γ_t
  2. Transformer cross-attention for B→C message passing
  3. Three-term loss: classification + reconstruction + modality alignment
  4. Interpretable output: γ_t heatmap reveals which cell types drive each bulk phenotype

Dependencies:
    pip install torch torch-geometric scanpy anndata scipy scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, softmax as pyg_softmax
import numpy as np
from typing import Optional, Dict, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH CONSTRUCTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_bulk_graph(
    bulk_expr: np.ndarray,       # (N_bulk, n_genes)
    gsva_scores: Optional[np.ndarray] = None,  # (N_bulk, n_pathways)
    threshold: float = 0.7,
) -> torch.Tensor:
    """
    Build intra-bulk adjacency from cosine similarity.
    Uses GSVA pathway scores if available (preferred — biologically meaningful edges).
    Falls back to raw expression cosine similarity.

    Returns edge_index: (2, E_BB)  in COO format for PyG
    """
    features = gsva_scores if gsva_scores is not None else bulk_expr
    features_t = torch.tensor(features, dtype=torch.float32)

    # L2-normalize for cosine similarity
    normed = F.normalize(features_t, dim=1)
    sim = normed @ normed.T                        # (N_bulk, N_bulk)
    sim.fill_diagonal_(0.0)

    # Threshold and convert to edge_index
    adj = (sim >= threshold).float()
    edge_index = adj.nonzero(as_tuple=False).T     # (2, E_BB)
    return edge_index, sim


def build_cross_modal_graph(
    bulk_shared: np.ndarray,     # (N_bulk, n_shared_genes)
    cell_shared: np.ndarray,     # (N_cell, n_shared_genes)
    top_k: int = 50,
    batch_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sparse bipartite B-C edges: top-K cells per bulk sample
    by cosine similarity on shared gene space.

    Processed in batches to avoid OOM with 536 x 63k matrix.

    Returns:
        edge_index_BC: (2, E_BC)  row=bulk_idx, col=cell_idx
        edge_weights:  (E_BC,)    cosine similarity scores
    """
    bulk_t = torch.tensor(bulk_shared, dtype=torch.float32)
    cell_t = torch.tensor(cell_shared, dtype=torch.float32)

    bulk_normed = F.normalize(bulk_t, dim=1)
    cell_normed = F.normalize(cell_t, dim=1)

    all_bulk_idx, all_cell_idx, all_weights = [], [], []
    N_bulk = bulk_normed.shape[0]

    for start in range(0, N_bulk, batch_size):
        end = min(start + batch_size, N_bulk)
        batch = bulk_normed[start:end]              # (batch, n_shared)
        sim_batch = batch @ cell_normed.T           # (batch, N_cell)

        topk_vals, topk_idx = sim_batch.topk(top_k, dim=1)  # (batch, K)

        for i, (b_i, b_off) in enumerate(zip(range(start, end), range(end - start))):
            all_bulk_idx.extend([b_i] * top_k)
            all_cell_idx.extend(topk_idx[b_off].tolist())
            all_weights.extend(topk_vals[b_off].tolist())

    edge_index = torch.tensor([all_bulk_idx, all_cell_idx], dtype=torch.long)
    weights    = torch.tensor(all_weights, dtype=torch.float32)
    return edge_index, weights


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODALITY ENCODERS
# ─────────────────────────────────────────────────────────────────────────────

class BulkEncoder(nn.Module):
    """
    Projects bulk expression (high-dim, low noise) into shared latent space.
    Input: raw log-normalized expression (N_bulk, n_genes)
    Output: latent embeddings (N_bulk, d_latent)
    """
    def __init__(self, n_genes: int, d_latent: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellEncoder(nn.Module):
    """
    Projects scRNA-seq expression (high-dim, sparse/noisy) into shared latent space.
    Higher dropout than bulk encoder to match the noisier input.
    Input: log-normalized expression (N_cell, n_genes)
    Output: latent embeddings (N_cell, d_latent)
    """
    def __init__(self, n_genes: int, d_latent: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellDecoder(nn.Module):
    """
    Reconstructs gene expression from cell latent embedding.
    Used for the reconstruction loss — forces h_C to retain expression info.
    """
    def __init__(self, d_latent: int, n_genes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, n_genes),
            nn.Softplus(),           # gene expression is non-negative
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-MODALITY ATTENTION  (B → C direction)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention_B2C(nn.Module):
    """
    Transformer-style cross-attention: cell queries, bulk keys/values.

    For each cell node i:
        q_i = W_q · h_Ci
        For each linked bulk k ∈ N_cross(i):
            k_k = W_k · h_Bk
            v_k = W_v · h_Bk
        β_ki = softmax( q_i · k_k / √d )
        message_i = Σ_k β_ki · v_k

    The attention weight β_ki is interpretable:
    "How much does bulk sample k's phenotype inform cell i's representation?"
    High β on MSI bulk = this cell is characteristic of MSI tumors.
    """

    def __init__(self, d_latent: int, n_heads: int = 4):
        super().__init__()
        self.d_head  = d_latent // n_heads
        self.n_heads = n_heads
        self.scale   = self.d_head ** -0.5

        self.W_q = nn.Linear(d_latent, d_latent, bias=False)
        self.W_k = nn.Linear(d_latent, d_latent, bias=False)
        self.W_v = nn.Linear(d_latent, d_latent, bias=False)
        self.W_o = nn.Linear(d_latent, d_latent, bias=False)

        self.norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        h_C: torch.Tensor,          # (N_cell_local, d)
        h_B: torch.Tensor,          # (N_bulk_local, d)
        edge_index_BC: torch.Tensor, # (2, E_BC)  [row=bulk, col=cell]
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sparse cross-attention — only over existing bipartite edges.
        Returns updated cell messages and optionally the attention weights.
        """
        N_cell = h_C.shape[0]
        bulk_idx = edge_index_BC[0]    # which bulk node
        cell_idx = edge_index_BC[1]    # which cell node

        # Project
        Q = self.W_q(h_C)[cell_idx]   # (E, d)
        K = self.W_k(h_B)[bulk_idx]   # (E, d)
        V = self.W_v(h_B)[bulk_idx]   # (E, d)

        # Scaled dot-product attention score per edge
        scores = (Q * K).sum(dim=-1) * self.scale   # (E,)

        # Softmax per cell (normalize over its linked bulk nodes)
        attn = pyg_softmax(scores, cell_idx, num_nodes=N_cell)  # (E,)

        # Weighted sum of bulk values → aggregated at each cell
        msg = attn.unsqueeze(-1) * V                # (E, d)
        agg = torch.zeros(N_cell, h_C.shape[1], device=h_C.device)
        agg.scatter_add_(0, cell_idx.unsqueeze(-1).expand_as(msg), msg)  # (N_cell, d)

        out = self.norm(self.W_o(agg) + h_C)        # residual connection

        if return_attn:
            return out, attn
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CELL-TYPE-AWARE C → B POOLING  (the novel message rule)
# ─────────────────────────────────────────────────────────────────────────────

class CellTypeAwarePooling_C2B(nn.Module):
    """
    Cell-type-aware aggregation of single-cell messages into bulk nodes.

    For each bulk sample i:
        1. Pool cell embeddings by cell type:
              p_t(i) = mean( h_Cc  for c ∈ N_cross(i) of type t )
        2. Learn how much each cell type matters for THIS bulk sample:
              γ_t(i) = softmax_t( W_γ · h_Bi )
        3. Aggregate:
              message_i = Σ_t  γ_t(i) · p_t(i)

    γ_t is the interpretable output — it reveals which cell types drive
    the bulk sample's latent representation. For MSI tumors, expect high
    γ on T-cell and immune populations.
    """

    def __init__(self, d_latent: int, n_cell_types: int, dropout: float = 0.1):
        super().__init__()
        self.n_cell_types = n_cell_types

        # Query: which cell types does THIS bulk sample care about?
        self.W_gamma = nn.Linear(d_latent, n_cell_types)

        # Per-type projection of cell embeddings before pooling
        self.W_cell_proj = nn.Linear(d_latent, d_latent)

        self.norm    = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_B: torch.Tensor,           # (N_bulk_local, d)
        h_C: torch.Tensor,           # (N_cell_local, d)
        edge_index_BC: torch.Tensor, # (2, E_BC)  [row=bulk, col=cell]
        cell_types: torch.Tensor,    # (N_cell_local,)  integer type labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            messages: (N_bulk_local, d)  updated bulk messages
            gamma:    (N_bulk_local, n_cell_types)  interpretable type attention
        """
        N_bulk  = h_B.shape[0]
        d       = h_B.shape[1]
        device  = h_B.device

        bulk_idx = edge_index_BC[0]    # (E,) — which bulk node each edge belongs to
        cell_idx = edge_index_BC[1]    # (E,) — which cell node

        # Project cell embeddings
        h_C_proj = self.W_cell_proj(h_C)   # (N_cell, d)

        # Step 1: Compute per-(bulk, cell_type) pooled embeddings
        # For each bulk node i, for each cell type t: mean of h_C_proj for linked cells of type t
        # Initialize: (N_bulk, n_cell_types, d)
        type_pools    = torch.zeros(N_bulk, self.n_cell_types, d, device=device)
        type_counts   = torch.zeros(N_bulk, self.n_cell_types, device=device)

        linked_types  = cell_types[cell_idx]              # (E,) — type of each linked cell
        linked_embs   = h_C_proj[cell_idx]                # (E, d) — embedding of each linked cell

        for t in range(self.n_cell_types):
            mask = (linked_types == t)                    # (E,) — edges to cells of type t
            if mask.sum() == 0:
                continue
            b_idx_t = bulk_idx[mask]                      # (E_t,) — which bulk nodes
            c_emb_t = linked_embs[mask]                   # (E_t, d)

            # Sum and count for mean
            type_pools[:, t].scatter_add_(0, b_idx_t.unsqueeze(-1).expand_as(c_emb_t), c_emb_t)
            type_counts[:, t].scatter_add_(0, b_idx_t, torch.ones(mask.sum(), device=device))

        # Normalize sums to means (avoid div by zero)
        counts_safe = type_counts.clamp(min=1).unsqueeze(-1)   # (N_bulk, n_types, 1)
        type_pools  = type_pools / counts_safe                 # (N_bulk, n_types, d)

        # Step 2: Learnable type-attention γ from bulk's own embedding
        gamma = F.softmax(self.W_gamma(h_B), dim=-1)           # (N_bulk, n_types)

        # Step 3: Weighted sum across cell types
        # gamma: (N_bulk, n_types, 1) * type_pools: (N_bulk, n_types, d) → sum over types
        messages = (gamma.unsqueeze(-1) * type_pools).sum(dim=1)  # (N_bulk, d)
        messages = self.dropout(messages)

        # Residual + norm
        out = self.norm(messages + h_B)

        return out, gamma   # gamma is the interpretable cell-type attention map


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ONE BIPARTITE GNN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class BipartiteGNNLayer(nn.Module):
    """
    One layer of heterogeneous message passing with three edge types.

    Operations per layer (in order):
        1. Intra-bulk GAT:          B ← B  (sample similarity graph)
        2. Intra-cell GAT:          C ← C  (cell kNN graph)
        3. Cross B→C attention:     C ← B  (bulk phenotype → cell)
        4. Cross C→B type-pooling:  B ← C  (cell decomposition → bulk)
        5. GRU update for both B and C nodes
    """

    def __init__(
        self,
        d_latent:     int,
        n_cell_types: int,
        n_heads_gat:  int = 4,
        n_heads_cross: int = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()

        # Intra-modality: standard GAT layers
        self.gat_bulk = GATConv(d_latent, d_latent // n_heads_gat,
                                heads=n_heads_gat, dropout=dropout, concat=True)
        self.gat_cell = GATConv(d_latent, d_latent // n_heads_gat,
                                heads=n_heads_gat, dropout=dropout, concat=True)

        # Cross-modal B→C
        self.cross_b2c = CrossModalAttention_B2C(d_latent, n_heads=n_heads_cross)

        # Cross-modal C→B (novel cell-type-aware pooling)
        self.cross_c2b = CellTypeAwarePooling_C2B(d_latent, n_cell_types, dropout)

        # GRU updaters — combine intra + cross messages
        self.gru_bulk = nn.GRUCell(d_latent * 2, d_latent)
        self.gru_cell = nn.GRUCell(d_latent * 2, d_latent)

        self.norm_B = nn.LayerNorm(d_latent)
        self.norm_C = nn.LayerNorm(d_latent)

    def forward(
        self,
        h_B: torch.Tensor,             # (N_bulk, d)
        h_C: torch.Tensor,             # (N_cell, d)
        edge_index_BB: torch.Tensor,   # (2, E_BB)
        edge_index_CC: torch.Tensor,   # (2, E_CC)
        edge_index_BC: torch.Tensor,   # (2, E_BC) [row=bulk, col=cell]
        cell_types: torch.Tensor,      # (N_cell,) int labels
        return_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # 1. Intra-bulk GAT
        m_BB = self.gat_bulk(h_B, edge_index_BB)       # (N_bulk, d)

        # 2. Intra-cell GAT
        m_CC = self.gat_cell(h_C, edge_index_CC)       # (N_cell, d)

        # 3. Cross B→C (cell receives from bulk)
        m_B2C, _ = self.cross_b2c(h_C, h_B, edge_index_BC,
                                   return_attn=False)  # (N_cell, d)

        # 4. Cross C→B with cell-type pooling (bulk receives from cells)
        m_C2B, gamma = self.cross_c2b(h_B, h_C, edge_index_BC,
                                       cell_types)     # (N_bulk, d), (N_bulk, n_types)

        # 5. GRU updates: concatenate intra + cross messages as input
        h_B_new = self.gru_bulk(
            torch.cat([m_BB, m_C2B], dim=-1),   # input: intra + cross
            h_B                                  # hidden: current state
        )
        h_C_new = self.gru_cell(
            torch.cat([m_CC, m_B2C], dim=-1),
            h_C
        )

        h_B_new = self.norm_B(h_B_new)
        h_C_new = self.norm_C(h_C_new)

        if return_gamma:
            return h_B_new, h_C_new, gamma
        return h_B_new, h_C_new, None


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL MODEL: BulkCellGNN
# ─────────────────────────────────────────────────────────────────────────────

class BulkCellGNN(nn.Module):
    """
    Full bipartite GNN for bulk-to-single-cell expression transfer.

    Pipeline:
        Bulk expression  →  BulkEncoder  →  h_B⁰
        Cell expression  →  CellEncoder  →  h_C⁰
        [BipartiteGNNLayer × n_layers]
        h_B^L  →  MLP classifier  →  MSI/MSS logits
        h_C^L  →  CellDecoder     →  reconstructed expression
                  (for reconstruction loss)

    Args:
        n_bulk_genes:  number of input genes for bulk (after shared gene filtering)
        n_cell_genes:  number of input genes for cells (HVGs)
        d_latent:      shared latent dimension
        n_classes:     classification output (2 for MSI/MSS, or more for CMS subtypes)
        n_cell_types:  number of known cell type categories in scRNA-seq data
        n_layers:      number of bipartite GNN layers (2 recommended)
        dropout:       dropout rate
    """

    def __init__(
        self,
        n_bulk_genes:  int,
        n_cell_genes:  int,
        d_latent:      int         = 256,
        n_classes:     int         = 2,
        n_cell_types:  int         = 9,   # GSE132465 has ~9 annotated types
        n_layers:      int         = 2,
        dropout:       float       = 0.3,
        cell_type_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.cell_type_names = cell_type_names or [f"Type_{i}" for i in range(n_cell_types)]

        # Modality-specific encoders
        self.bulk_encoder = BulkEncoder(n_bulk_genes, d_latent, dropout)
        self.cell_encoder = CellEncoder(n_cell_genes, d_latent, dropout)

        # Bipartite GNN layers
        self.gnn_layers = nn.ModuleList([
            BipartiteGNNLayer(d_latent, n_cell_types, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.classifier = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent // 2, n_classes),
        )

        self.cell_decoder = CellDecoder(d_latent, n_cell_genes, dropout)

    def forward(
        self,
        bulk_x:        torch.Tensor,    # (N_bulk, n_bulk_genes)
        cell_x:        torch.Tensor,    # (N_cell, n_cell_genes)
        edge_index_BB: torch.Tensor,    # (2, E_BB)
        edge_index_CC: torch.Tensor,    # (2, E_CC)
        edge_index_BC: torch.Tensor,    # (2, E_BC)
        cell_types:    torch.Tensor,    # (N_cell,) int type labels
        return_gamma:  bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full bipartite GNN.

        Returns dict with:
            'logits':     (N_bulk, n_classes)      — for classification loss
            'cell_recon': (N_cell, n_cell_genes)   — for reconstruction loss
            'h_B':        (N_bulk, d_latent)        — final bulk embeddings
            'h_C':        (N_cell, d_latent)        — final cell embeddings
            'gamma':      (N_bulk, n_cell_types)    — interpretable type attention (if requested)
        """
        # Initial encodings
        h_B = self.bulk_encoder(bulk_x)    # (N_bulk, d)
        h_C = self.cell_encoder(cell_x)    # (N_cell, d)

        gamma_last = None

        # Iterative message passing
        for l, layer in enumerate(self.gnn_layers):
            is_last = (l == self.n_layers - 1)
            h_B, h_C, gamma = layer(
                h_B, h_C,
                edge_index_BB, edge_index_CC, edge_index_BC,
                cell_types,
                return_gamma=(return_gamma and is_last),
            )
            if is_last:
                gamma_last = gamma

        # Output heads
        logits     = self.classifier(h_B)          # (N_bulk, n_classes)
        cell_recon = self.cell_decoder(h_C)         # (N_cell, n_cell_genes)

        out = {
            'logits':     logits,
            'cell_recon': cell_recon,
            'h_B':        h_B,
            'h_C':        h_C,
        }
        if return_gamma:
            out['gamma'] = gamma_last
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MULTI-TASK LOSS
# ─────────────────────────────────────────────────────────────────────────────

class BulkCellLoss(nn.Module):
    """
    Three-term loss function:

    L_total = λ_cls · L_classify
            + λ_rec · L_reconstruct
            + λ_aln · L_align

    L_classify:   cross-entropy on bulk MSI/MSS labels
    L_reconstruct: MSE between decoded and original cell expression
                   (forces h_C to retain expression information, not just
                    become useful for bulk classification)
    L_align:       MSE between bulk embedding and mean of its linked cell embeddings
                   (regularizes cross-modality edge quality — bulk should be
                    "near" the cells that compose it in latent space)
    """

    def __init__(self, lambda_cls=1.0, lambda_rec=0.5, lambda_aln=0.1):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_aln = lambda_aln

    def forward(
        self,
        logits:        torch.Tensor,    # (N_bulk, n_classes)
        labels:        torch.Tensor,    # (N_bulk,) int
        cell_recon:    torch.Tensor,    # (N_cell, n_genes)
        cell_x:        torch.Tensor,    # (N_cell, n_genes)  original
        h_B:           torch.Tensor,    # (N_bulk, d)
        h_C:           torch.Tensor,    # (N_cell, d)
        edge_index_BC: torch.Tensor,    # (2, E_BC)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # 1. Classification loss
        L_cls = F.cross_entropy(logits, labels)

        # 2. Cell reconstruction loss (MSE on log-normalized expression)
        L_rec = F.mse_loss(cell_recon, cell_x)

        # 3. Alignment loss: h_B should be close to mean(h_C) for linked cells
        N_bulk  = h_B.shape[0]
        d       = h_B.shape[1]
        bulk_idx = edge_index_BC[0]
        cell_idx = edge_index_BC[1]

        # Mean of linked cell embeddings per bulk node
        cell_means = torch.zeros(N_bulk, d, device=h_B.device)
        counts     = torch.zeros(N_bulk, device=h_B.device)
        cell_means.scatter_add_(0, bulk_idx.unsqueeze(-1).expand(-1, d), h_C[cell_idx])
        counts.scatter_add_(0, bulk_idx, torch.ones(bulk_idx.shape[0], device=h_B.device))
        counts = counts.clamp(min=1).unsqueeze(-1)
        cell_means = cell_means / counts   # (N_bulk, d)

        # Only penalize bulk nodes that have linked cells
        has_cells  = (counts.squeeze(-1) > 0)
        L_aln = F.mse_loss(h_B[has_cells], cell_means[has_cells].detach())
        # Note: detach cell_means — don't let alignment loss pull cells toward bulk,
        # only pull bulk toward cells. This prevents mode collapse.

        L_total = (self.lambda_cls * L_cls
                 + self.lambda_rec * L_rec
                 + self.lambda_aln * L_aln)

        losses = {
            'total':    L_total.item(),
            'classify': L_cls.item(),
            'recon':    L_rec.item(),
            'align':    L_aln.item(),
        }
        return L_total, losses


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bulk_classification(model, dataloader, device='cpu'):
    """
    Compute AUC-ROC for MSI/MSS classification on bulk samples.
    Compare against your SVM baseline (AUC=0.964 on GSE39582).
    """
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            out = model(**batch)
            probs = F.softmax(out['logits'], dim=-1)[:, 1]  # P(MSI)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    return roc_auc_score(all_labels, all_probs)


def evaluate_cell_clustering(h_C, cell_type_labels, n_types):
    """
    Evaluate whether h_C clusters by known cell type.
    Uses k-means on GNN embeddings, compares to ground-truth labels.
    Metric: Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    h = h_C.detach().cpu().numpy()
    km = KMeans(n_clusters=n_types, random_state=42, n_init=10)
    pred = km.fit_predict(h)

    ari = adjusted_rand_score(cell_type_labels, pred)
    nmi = normalized_mutual_info_score(cell_type_labels, pred)
    return {'ARI': ari, 'NMI': nmi}


def plot_gamma_heatmap(
    gamma: torch.Tensor,          # (N_bulk, n_cell_types)
    msi_labels: np.ndarray,       # (N_bulk,)  0=MSS, 1=MSI
    cell_type_names: List[str],
    save_path: str = 'gamma_heatmap.png',
):
    """
    The interpretability output of BulkCell-GNN.

    Plots mean γ_t per cell type, split by MSI vs MSS bulk samples.
    Expected biological finding:
      - MSI: high γ on T cells, NK cells (immune-hot TME)
      - MSS: high γ on epithelial, stromal cells (immune-cold TME)

    This is directly validatable against known CRC TME biology.
    """
    import matplotlib.pyplot as plt

    gamma_np = gamma.detach().cpu().numpy()
    msi_mask = (msi_labels == 1)
    mss_mask = (msi_labels == 0)

    mean_gamma_msi = gamma_np[msi_mask].mean(0)    # (n_types,)
    mean_gamma_mss = gamma_np[mss_mask].mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, vals, title, color in [
        (axes[0], mean_gamma_msi, 'MSI samples', '#1D9E75'),
        (axes[1], mean_gamma_mss, 'MSS samples', '#7F77DD'),
    ]:
        bars = ax.barh(cell_type_names, vals, color=color, alpha=0.8)
        ax.set_xlabel('Mean cell-type attention γ_t')
        ax.set_title(title)
        ax.axvline(1 / len(cell_type_names), color='gray', linestyle='--',
                   alpha=0.5, label='uniform baseline')
        ax.legend(fontsize=8)

    plt.suptitle('Cell-type attention γ_t by MSI/MSS phenotype\n'
                 '(high γ = this cell type strongly shapes bulk representation)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved gamma heatmap → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: BulkCellGNN,
    bulk_x: torch.Tensor,        # (N_bulk, n_bulk_genes)
    cell_x: torch.Tensor,        # (N_cell, n_cell_genes)
    labels: torch.Tensor,        # (N_bulk,)  MSI/MSS labels
    cell_types: torch.Tensor,    # (N_cell,)  int type labels
    edge_index_BB: torch.Tensor,
    edge_index_CC: torch.Tensor,
    edge_index_BC: torch.Tensor,
    train_mask: torch.Tensor,    # (N_bulk,) bool — train/val split
    val_mask:   torch.Tensor,
    n_epochs:   int   = 100,
    lr:         float = 1e-3,
    device:     str   = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Full-graph training (all bulk + their linked cells per forward pass).
    For 536 bulk samples this fits in memory; scale to mini-batching if needed.
    """
    model = model.to(device)
    criterion = BulkCellLoss(lambda_cls=1.0, lambda_rec=0.5, lambda_aln=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Move all tensors to device
    bulk_x       = bulk_x.to(device)
    cell_x       = cell_x.to(device)
    labels       = labels.to(device)
    cell_types   = cell_types.to(device)
    edge_index_BB = edge_index_BB.to(device)
    edge_index_CC = edge_index_CC.to(device)
    edge_index_BC = edge_index_BC.to(device)

    best_val_auc = 0.0
    from sklearn.metrics import roc_auc_score

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(bulk_x, cell_x,
                    edge_index_BB, edge_index_CC, edge_index_BC,
                    cell_types)

        L_total, loss_dict = criterion(
            out['logits'][train_mask], labels[train_mask],
            out['cell_recon'], cell_x,
            out['h_B'], out['h_C'],
            edge_index_BC,
        )

        L_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out_val = model(bulk_x, cell_x,
                                edge_index_BB, edge_index_CC, edge_index_BC,
                                cell_types, return_gamma=True)

            val_probs = F.softmax(out_val['logits'][val_mask], dim=-1)[:, 1].cpu().numpy()
            val_labels = labels[val_mask].cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'bulkcell_gnn_best.pt')

            print(f"Epoch {epoch:3d} | "
                  f"L={loss_dict['total']:.4f}  "
                  f"cls={loss_dict['classify']:.4f}  "
                  f"rec={loss_dict['recon']:.4f}  "
                  f"aln={loss_dict['align']:.4f}  "
                  f"val_AUC={val_auc:.4f}")

    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    return best_val_auc


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MINIMAL SMOKE TEST (no real data)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cpu'

    # ── Toy dimensions ──────────────────────────────────────────────────────
    N_BULK      = 50      # → 536 in real run (GSE39582)
    N_CELL      = 500     # → 63,689 in real run (GSE132465)
    N_BULK_GENES = 5000   # shared gene set after HVG intersection
    N_CELL_GENES = 5000
    D_LATENT    = 128
    N_CELL_TYPES = 9      # GSE132465 annotated cell types
    N_CLASSES   = 2       # MSI / MSS
    K_CROSS     = 10      # top-K cross-modal edges (50 in real run)

    # ── Fake data ────────────────────────────────────────────────────────────
    bulk_x     = torch.randn(N_BULK, N_BULK_GENES).abs()      # log-norm expression
    cell_x     = torch.randn(N_CELL, N_CELL_GENES).abs()
    labels     = torch.randint(0, N_CLASSES, (N_BULK,))
    cell_types = torch.randint(0, N_CELL_TYPES, (N_CELL,))

    # ── Fake graphs ──────────────────────────────────────────────────────────
    # Intra-bulk: random sparse graph
    bb_src = torch.randint(0, N_BULK, (N_BULK * 3,))
    bb_dst = torch.randint(0, N_BULK, (N_BULK * 3,))
    edge_index_BB = torch.stack([bb_src, bb_dst])

    # Intra-cell: kNN-style
    cc_src = torch.randint(0, N_CELL, (N_CELL * 15,))
    cc_dst = torch.randint(0, N_CELL, (N_CELL * 15,))
    edge_index_CC = torch.stack([cc_src, cc_dst])

    # Cross-modal: top-K_CROSS cells per bulk sample
    bc_bulk = torch.arange(N_BULK).repeat_interleave(K_CROSS)
    bc_cell = torch.randint(0, N_CELL, (N_BULK * K_CROSS,))
    edge_index_BC = torch.stack([bc_bulk, bc_cell])

    # ── Instantiate model ────────────────────────────────────────────────────
    model = BulkCellGNN(
        n_bulk_genes   = N_BULK_GENES,
        n_cell_genes   = N_CELL_GENES,
        d_latent       = D_LATENT,
        n_classes      = N_CLASSES,
        n_cell_types   = N_CELL_TYPES,
        n_layers       = 2,
        dropout        = 0.2,
        cell_type_names = ['T cell', 'CD8 T', 'Macrophage', 'B cell',
                           'NK cell', 'Epithelial', 'Stromal', 'Myeloid', 'DC'],
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    out = model(bulk_x, cell_x,
                edge_index_BB, edge_index_CC, edge_index_BC,
                cell_types, return_gamma=True)

    print(f"Logits shape:     {out['logits'].shape}")
    print(f"Cell recon shape: {out['cell_recon'].shape}")
    print(f"h_B shape:        {out['h_B'].shape}")
    print(f"h_C shape:        {out['h_C'].shape}")
    print(f"Gamma shape:      {out['gamma'].shape}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = BulkCellLoss()
    train_mask = torch.ones(N_BULK, dtype=torch.bool)
    L, losses = criterion(out['logits'], labels, out['cell_recon'], cell_x,
                          out['h_B'], out['h_C'], edge_index_BC)
    print(f"\nLoss breakdown:")
    for k, v in losses.items():
        print(f"  {k:12s}: {v:.4f}")

    # ── Verify gamma interpretability ────────────────────────────────────────
    gamma = out['gamma']
    print(f"\nGamma (cell-type attention) per bulk sample — sums to 1.0:")
    print(f"  Row sums (should all be ~1): {gamma.sum(dim=-1)[:5].detach().numpy().round(3)}")

    print("\nSmoke test passed. Ready to run on GSE39582 + GSE132465.")
    print("\nNext steps:")
    print("  1. Load GSE39582 with GEOparse, extract log-normalized expression + MSI/MSS labels")
    print("  2. Load GSE132465 with scanpy, run QC + HVG selection + cell type annotation")
    print("  3. Intersect gene sets → build_cross_modal_graph()")
    print("  4. Replace toy tensors above with real data and run train()")
    print("  5. After training: plot_gamma_heatmap() for XAI interpretation")
