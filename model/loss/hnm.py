"""HNM radius and cone adapter for the original HyperAMA loss interface."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .hyperbolic import ContrastiveGeometry


def paper_contrastive_loss(logits):
    """HNM Eqs. (8)-(9): negative-only denominators, batch sums, two directions."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or logits.shape[0] < 2:
        raise ValueError("HNM paper contrastive loss requires a square batch with at least two pairs")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite HNM contrastive logits")
    diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    negatives = logits.masked_fill(diagonal, -torch.inf)
    positive = logits.diagonal()
    forward = (torch.logsumexp(negatives, dim=1)-positive).sum()
    reverse = (torch.logsumexp(negatives, dim=0)-positive).sum()
    return (forward+reverse)/2


class HNMPaperContrastiveLoss(nn.Module):
    """Literal paper objective; original AMA metadata does not mask negatives here."""
    def __init__(self, geometry, initial_temperature=0.07, minimum_temperature=0.01):
        super().__init__()
        if (not math.isfinite(initial_temperature) or not math.isfinite(minimum_temperature)
                or not 0 < minimum_temperature <= initial_temperature):
            raise ValueError("HNM temperature must start at or above its positive lower bound")
        self.geometry = geometry
        self.minimum_temperature = float(minimum_temperature)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.settings = {"objective": "hnm_eq8_eq9_batch_sum",
                         "initial_temperature": float(initial_temperature),
                         "minimum_temperature": self.minimum_temperature}

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp_min(self.minimum_temperature)

    def get_extra_state(self):
        return dict(self.settings)

    def set_extra_state(self, state):
        if state != self.settings:
            raise ValueError("HNM contrastive checkpoint settings differ; use the saved configuration")

    def forward(self, preds, targets, **metadata):
        # Accept the unchanged FACED/SEED call signatures. Eq. (8) keeps every k != i.
        with torch.autocast(device_type=preds.device.type, enabled=False):
            return paper_contrastive_loss(self.geometry(preds, targets)/self.temperature)


def radius_floor(curvature, cone_k, aperture_max_deg):
    return math.asinh(2 * cone_k / math.sin(math.radians(aperture_max_deg))) / math.sqrt(curvature)


def cone_terms(parent_u, parent_rho, child_u, child_rho, curvature, cone_k,
               tangent_epsilon=1e-10):
    """Angle from the outward parent axis to its paired EEG child geodesic."""
    with torch.autocast(device_type=parent_u.device.type, enabled=False):
        u, v = F.normalize(parent_u.double(), dim=-1), F.normalize(child_u.double(), dim=-1)
        a, b = math.sqrt(curvature)*parent_rho.double(), math.sqrt(curvature)*child_rho.double()
        t = (u*v).sum(-1)
        if not torch.isfinite(t).all() or (t.abs() > 1+1e-12).any():
            raise ValueError("Invalid direction cosine in HNM cone")
        t = t.clamp(-1, 1)
        along = torch.sinh(b-a)-torch.cosh(a)*torch.sinh(b)*(1-t)
        across = torch.sinh(b)*torch.linalg.vector_norm(v-t[:, None]*u, dim=-1)
        length = torch.hypot(along, across)
        if not torch.isfinite(length).all():
            raise ValueError("Non-finite HNM cone tangent")
        ratio = 2*cone_k/torch.sinh(a)
        if not torch.isfinite(ratio).all() or ((ratio <= 0) | (ratio >= 1)).any():
            raise ValueError("Material radius outside the HNM cone domain")
        # Smooth the transverse component to keep atan2 and its gradient defined.
        # Coincidence receives a pi/2 surrogate angle, not a satisfied cone.
        safe_across = torch.hypot(across, torch.full_like(across, tangent_epsilon))
        theta, psi = torch.atan2(safe_across, along), torch.asin(ratio)
        return theta, psi, F.relu(theta-psi)


class HNMGeometry(ContrastiveGeometry):
    """Keep normalization, expmap0 and dist; add one radius head per modality."""
    def __init__(self, curvature, options, dim=1024):
        super().__init__("hyperbolic", curvature)
        self.curvature = float(curvature)
        self.dim = int(dim)
        radius, cone = options["radius"], options["cone"]
        self.settings = {
            "spec_id": str(options["spec_id"]), "curvature": self.curvature, "dim": self.dim,
            **{f"radius_{key}": float(radius[key]) for key in (
                "cone_k", "aperture_max_deg", "maximum", "initial", "minimum_direction_norm")},
            **{f"cone_{key}": float(cone[key]) for key in (
                "weight", "tangent_epsilon")},
        }
        s = self.settings
        if not all(math.isfinite(value) for value in s.values() if isinstance(value, (float, int))):
            raise ValueError("HNM settings must be finite")
        if s["radius_cone_k"] <= 0 or not 0 < s["radius_aperture_max_deg"] < 90:
            raise ValueError("HNM requires K > 0 and an aperture in (0, 90) degrees")
        self.rho_min = radius_floor(curvature, s["radius_cone_k"], s["radius_aperture_max_deg"])
        self.rho_max = s["radius_maximum"]
        initial = s["radius_initial"]
        if not 0 < self.rho_min < initial < self.rho_max:
            raise ValueError("HNM radius initialization must lie strictly inside its bounds")
        if (s["radius_minimum_direction_norm"] <= 0 or s["cone_tangent_epsilon"] <= 0
                or s["cone_weight"] < 0):
            raise ValueError("Invalid HNM weight or tolerances")
        # Additional heads must not advance the original model/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.eeg_radius = nn.Linear(self.dim, 1)
            self.material_radius = nn.Linear(self.dim, 1)
            for head in (self.eeg_radius, self.material_radius):
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, math.log((initial-self.rho_min)/(self.rho_max-initial)))

    def get_extra_state(self):
        return dict(self.settings)

    def set_extra_state(self, state):
        if state != self.settings:
            raise ValueError("HNM checkpoint settings differ; restore with the saved HNM configuration")

    def _polar(self, features, head):
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.reshape(features.shape[0], -1).float()
            if features.shape[1] != self.dim:
                raise ValueError(f"HNM expects the original {self.dim}-dimensional embeddings")
            norms = torch.linalg.vector_norm(features, dim=1)
            if (not torch.isfinite(features).all()
                    or (norms <= self.settings["radius_minimum_direction_norm"]).any()):
                raise ValueError("HNM direction is non-finite or has vanishing norm")
            direction = self._normalize(features)
            rho = self.rho_min + (self.rho_max-self.rho_min)*torch.sigmoid(head(direction))
            point = self.ball.expmap0(direction * (rho / 2))
            return point, direction, rho.squeeze(-1)

    def project(self, features, branch="eeg"):
        if branch not in ("eeg", "material"):
            raise ValueError("HNM branch must be eeg or material")
        head = self.eeg_radius if branch == "eeg" else self.material_radius
        return self._polar(features, head)[0]

    def forward(self, eeg, targets):
        if eeg.shape[0] != targets.shape[0]:
            raise ValueError("EEG and targets must have the same batch size")
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            eeg, targets = self.project(eeg), self.project(targets, "material")
            return -self.ball.dist(eeg[:, None, :], targets[None, :, :])

    def hnm_loss(self, eeg, targets, epoch):
        # Keep epoch in the existing Hyper hook signature; the cone weight is constant.
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            child, child_u, child_rho = self._polar(eeg, self.eeg_radius)
            parent, parent_u, parent_rho = self._polar(targets, self.material_radius)
            distance = self.ball.dist(parent, child)
            s = self.settings
            weight = s["cone_weight"]
            cone = eeg.float().sum()*0.0
            if weight > 0:
                cone = cone_terms(parent_u, parent_rho, child_u, child_rho, self.curvature,
                                  s["radius_cone_k"], s["cone_tangent_epsilon"])[2].mean().float()
            total = weight*cone
            diagnostics = {"cone": cone.detach(), "cone_weight": weight,
                           "weighted": total.detach(),
                           "radius_material": parent_rho.detach().mean(),
                           "radius_eeg": child_rho.detach().mean(), "pair_distance": distance.detach().mean()}
            return total, diagnostics
