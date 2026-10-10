"""HNM math and regression tests for reuse of the original HyperAMA pipeline."""
import math
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F

from model import get_extractor_class
from model.loss.con_loss import MultiModalInfoNCELoss
from model.loss.seed_con_loss import MultiModalInfoNCELoss as SeedAlignment
from model.loss.hyperbolic import ContrastiveGeometry, MultiScaleContrastiveLoss
from model.loss.hnm import (HNMGeometry, HNMPaperContrastiveLoss, paper_contrastive_loss,
                            cone_terms)

ROOT = Path(__file__).resolve().parents[1]


def config(experiment="hnm_h2", dataset="FACED"):
    with initialize_config_dir(config_dir=str(ROOT / "cfgs"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"data={dataset}", f"+experiment={experiment}"])


def geometry(dim=8):
    cfg = config()
    return HNMGeometry(cfg.train.contrastive.hyperbolic.curvature, cfg.train.hnm, dim)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1024, 1024)

    def forward(self, x, **kwargs):
        return self.net(x)

    def set_saveFea(self, flag):
        pass


def model(cfg, backbone=None):
    module = get_extractor_class(cfg.data.dataset_name)(backbone or ToyBackbone(), cfg.train)
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None
    module.optimizers = lambda: type("Optimizer", (), {"param_groups": [{"lr": .0007}]})()
    return module


class HNMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_paper_contrastive_matches_literal_equations_and_gradients(self):
        scores = torch.tensor([[-.2, -1.1, -.8], [-.6, -.3, -1.5], [-1.4, -.9, -.1]],
                              dtype=torch.float64, requires_grad=True)
        terms = []
        for direction in (scores, scores.T):
            for i in range(3):
                denominator = sum(direction[i, k].exp() for k in range(3) if k != i)
                terms.append(-torch.log(direction[i, i].exp()/denominator))
        expected = sum(terms)/2
        actual = paper_contrastive_loss(scores)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, scores)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(actual_grad.diagonal(), -torch.ones(3, dtype=torch.float64))
        # Identical logits give N*log(N-1), not batch-mean log(N-1) or CE log(N).
        torch.testing.assert_close(paper_contrastive_loss(torch.zeros(4, 4)),
                                   torch.tensor(4*math.log(3)))
        self.assertLess(actual.item(), 0)  # The paper objective is allowed to be negative.

    def test_paper_contrastive_extreme_logits_and_invalid_batches(self):
        scores = torch.tensor([[-10000., -10003., -10005.],
                               [-10006., -10001., -10004.],
                               [-10007., -10002., -10000.]], requires_grad=True)
        loss = paper_contrastive_loss(scores)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        for shape in ((0, 0), (1, 1), (2, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "at least two"):
                paper_contrastive_loss(torch.zeros(shape))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            paper_contrastive_loss(torch.full((2, 2), float("nan")))

    def test_paper_keeps_all_off_diagonal_pairs_and_learns_temperature(self):
        geo = geometry()
        criterion = HNMPaperContrastiveLoss(geo)
        torch.manual_seed(13)
        x, y = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
        plain = criterion(x, y)
        # All samples share material/content IDs; the paper still includes k != i.
        metadata = {"vid_ids": torch.zeros(4), "sub_ids": torch.arange(4),
                    "content_ids": torch.zeros(4), "original_targets": y}
        torch.testing.assert_close(criterion(x, y, **metadata), plain)
        expected = paper_contrastive_loss(geo(x, y)/criterion.temperature)
        torch.testing.assert_close(plain, expected)
        optimizer = torch.optim.Adam(criterion.parameters(), lr=config().train.lr)
        before = criterion.log_temperature.detach().clone()
        plain.backward()
        for value in (x, y, criterion.log_temperature, geo.eeg_radius.weight, geo.material_radius.weight):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, criterion.log_temperature.detach()))
        with torch.no_grad():
            criterion.log_temperature.fill_(math.log(.001))
        self.assertAlmostEqual(criterion.temperature.item(), .01)
        self.assertTrue(torch.isfinite(criterion(x, y)))

    def test_paper_route_for_all_original_modes_and_datasets(self):
        for dataset in ("FACED", "FACED_def_c2", "SEED"):
            for mode in (0, 1, 2, 3):
                with self.subTest(dataset=dataset, mode=mode):
                    cfg = config(dataset=dataset)
                    cfg.train.pretrain_mode = mode
                    cfg.train.use_two_stage_projector = False
                    module = model(cfg)
                    criterion = module.distill_criterion if mode in (2, 3) else module.criterion
                    self.assertIs(type(criterion), HNMPaperContrastiveLoss)
                    self.assertIs(criterion.geometry, module.contrastive_geometry)
                    self.assertAlmostEqual(criterion.temperature.item(), .07)
                    optimizer = module.configure_optimizers()["optimizer"]
                    parameters = [p for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(sum(p is criterion.log_temperature for p in parameters), 1)

    def test_existing_hyperparameters_unchanged(self):
        for dataset in ("FACED", "FACED_def_c2", "SEED"):
            base = config("hyper_distance", dataset)
            added = config(dataset=dataset)
            added.log = base.log
            before, after = (OmegaConf.to_container(x, resolve=True) for x in (base, added))
            after["train"].pop("hnm")
            self.assertEqual(before, after)

    def test_initial_mapping_equals_original_and_rng_is_preserved(self):
        state = torch.get_rng_state().clone()
        geo = geometry(1024)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in geo.parameters() if p.requires_grad), 2050)
        self.assertAlmostEqual(geo.rho_min, 1.2333999783875633)
        x, y = torch.randn(4, 1024), torch.randn(4, 1024)
        old = ContrastiveGeometry()
        torch.testing.assert_close(geo.project(x), old.project(x))
        torch.testing.assert_close(geo.project(y, "material"), old.project(y))
        torch.testing.assert_close(geo(x, y), old(x, y))
        expected = -geo.ball.dist(geo.project(x)[:, None, :], geo.project(y, "material")[None, :, :])
        torch.testing.assert_close(geo(x, y), expected, rtol=0, atol=0)

    def test_same_mapping_inside_original_nce_and_auxiliary(self):
        geo = geometry()
        with torch.no_grad():
            geo.eeg_radius.weight.normal_()
            geo.material_radius.weight.normal_()
        x, y = torch.randn(6, 8)*5, torch.randn(6, 8)*3
        torch.testing.assert_close(geo(x, y), geo(F.normalize(x, dim=1), F.normalize(y, dim=1)))
        loss, metrics = geo.hnm_loss(x, y, 8)
        torch.testing.assert_close(metrics["pair_distance"], -geo(x, y).diagonal().mean())
        self.assertTrue(torch.isfinite(loss))

    def test_original_masks_and_emotion_scale_reused_with_learned_radii(self):
        geo = geometry()
        with torch.no_grad():
            geo.eeg_radius.weight.normal_()
            geo.material_radius.weight.normal_()
        x, y = torch.randn(6, 8), torch.randn(6, 8)
        vids, subs = torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 0, 0, 1, 1, 1])
        scores = geo(x, y)/.07
        mask = vids[:, None].eq(vids[None, :]) & ~subs[:, None].eq(subs[None, :])
        scores = scores.masked_fill(mask, -1e4)
        expected = (F.cross_entropy(scores, torch.arange(6)) + F.cross_entropy(scores.T, torch.arange(6)))/2
        faced, seed = MultiModalInfoNCELoss(temperature=.07), SeedAlignment(temperature=.07)
        faced.geometry = seed.geometry = geo
        torch.testing.assert_close(faced(x, y, vid_ids=vids, sub_ids=subs), expected)
        torch.testing.assert_close(seed(x, y, content_ids=vids), expected)
        scale = MultiScaleContrastiveLoss(geo, {"emotion": {"enabled": True, "weight": .1}})
        value, parts = scale(x, y, labels=vids)
        torch.testing.assert_close(value, .1*parts["emotion"])
        self.assertIs(scale.geometry, geo)

    def test_radius_bounds_gradients_and_original_geoopt_distance(self):
        geo = geometry()
        x, y = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
        loss = -geo(x, y).mean()+geo.hnm_loss(x, y, 8)[0]
        loss.backward()
        for value in (x, y, geo.eeg_radius.weight, geo.material_radius.weight):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum(), 0)
        for bias in (-100., 100.):
            with torch.no_grad():
                geo.eeg_radius.bias.fill_(bias)
                geo.material_radius.bias.fill_(bias)
            points, _, rho = geo._polar(x, geo.eeg_radius)
            self.assertTrue(((rho >= geo.rho_min) & (rho <= geo.rho_max)).all())
            torch.testing.assert_close(geo.ball.dist0(points), rho, atol=1e-6, rtol=1e-6)
            self.assertLessEqual((-geo(x, y)).max().item(), 2*geo.rho_max+1e-5)

    def test_cone_against_lorentz_tangent_oracle(self):
        torch.manual_seed(2)
        u, v = F.normalize(torch.randn(4, 8, dtype=torch.float64), dim=1), F.normalize(torch.randn(4, 8, dtype=torch.float64), dim=1)
        rp, rc = torch.full((4,), 1.6, dtype=torch.float64), torch.full((4,), 2.3, dtype=torch.float64)
        root_c = math.sqrt(.1)
        a, b = root_c*rp, root_c*rc
        parent = torch.cat((a.cosh()[:, None], a.sinh()[:, None]*u), dim=1)
        child = torch.cat((b.cosh()[:, None], b.sinh()[:, None]*v), dim=1)
        axis = torch.cat((a.sinh()[:, None], a.cosh()[:, None]*u), dim=1)
        def minkowski(x, y):
            return -x[:, 0]*y[:, 0]+(x[:, 1:]*y[:, 1:]).sum(1)
        tangent = child+minkowski(parent, child)[:, None]*parent
        expected = torch.acos(minkowski(axis, tangent)/minkowski(tangent, tangent).sqrt())
        theta, psi, violation = cone_terms(u, rp, v, rc, .1, .1)
        torch.testing.assert_close(theta, expected)
        torch.testing.assert_close(psi, torch.asin(.2/torch.sinh(a)))
        torch.testing.assert_close(violation, (expected-psi).relu())
        self.assertTrue(torch.autograd.gradcheck(
            lambda p, c: cone_terms(u, p, v, c, .1, .1)[2],
            (rp.requires_grad_(), rc.requires_grad_())))

    def test_outward_inward_coincident_and_constant_cone_weight(self):
        u = torch.tensor([[1., 0.]], dtype=torch.float64)
        rp, rc = torch.tensor([1.6]), torch.tensor([2.3])
        self.assertAlmostEqual(cone_terms(u, rp, u, rc, .1, .1)[0].item(), 0.)
        self.assertAlmostEqual(cone_terms(u, rc, u, rp, .1, .1)[0].item(), math.pi)
        theta, psi, violation = cone_terms(u, rp, u, rp, .1, .1)
        self.assertAlmostEqual(theta.item(), math.pi/2)
        self.assertGreater(violation.item(), 0)
        geo = geometry()
        x = torch.randn(2, 8)
        first, metrics = geo.hnm_loss(x, x, 0)
        self.assertTrue(torch.isfinite(first))
        self.assertGreater(first.item(), 0.)
        self.assertEqual(metrics["cone_weight"], .01)
        for epoch in (1, 2, 3, 7, 20):
            with self.subTest(epoch=epoch):
                loss, metrics = geo.hnm_loss(x, x, epoch)
                self.assertEqual(metrics["cone_weight"], .01)
                torch.testing.assert_close(loss, first, rtol=0, atol=0)
        self.assertNotIn("zero_epochs", config().train.hnm.cone)
        self.assertNotIn("ramp_epochs", config().train.hnm.cone)

    def test_near_coincident_cones_have_finite_gradients(self):
        for angle in (0., 1e-12, 1e-10, 1e-8, 1e-6):
            for gap in (-1e-10, 0., 1e-10):
                with self.subTest(angle=angle, gap=gap):
                    u = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
                    v = torch.tensor([[math.cos(angle), math.sin(angle)]],
                                     dtype=torch.float64, requires_grad=True)
                    rp = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
                    rc = torch.tensor([2.+gap], dtype=torch.float64, requires_grad=True)
                    theta, psi, violation = cone_terms(u, rp, v, rc, .1, .1)
                    self.assertTrue(all(torch.isfinite(x).all() for x in (theta, psi, violation)))
                    violation.sum().backward()
                    self.assertTrue(all(torch.isfinite(x.grad).all() for x in (u, v, rp, rc)))
                    if angle == 0. and gap == 0.:
                        self.assertGreater(violation.item(), 0.)
                        self.assertGreater(rp.grad.item(), 0.)
                        self.assertLess(rc.grad.item(), 0.)

    def test_coincident_hnm_loss_backward_and_radius_update(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                geo = geometry()
                x = torch.zeros(2, 8, dtype=dtype)
                x[:, 0] = 1
                x.requires_grad_()
                y = x.detach().clone().requires_grad_()
                optimizer = torch.optim.Adam(geo.parameters(), lr=config().train.lr)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    loss, metrics = geo.hnm_loss(x, y, 8)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics["cone"].item(), 0.)
                loss.backward()
                self.assertTrue(all(torch.isfinite(t.grad).all() for t in (x, y)))
                for head in (geo.material_radius, geo.eeg_radius):
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
                self.assertGreater(geo.material_radius.bias.grad.item(), 0.)
                self.assertLess(geo.eeg_radius.bias.grad.item(), 0.)
                torch.nn.utils.clip_grad_norm_(geo.parameters(), 1.0)
                optimizer.step()
                _, updated = geo.hnm_loss(x, y, 8)
                self.assertGreater(updated["radius_eeg"].item(), updated["radius_material"].item())

    def test_cone_still_rejects_nonfinite_pairs(self):
        u = torch.tensor([[1., 0.]], dtype=torch.float64)
        rp = torch.tensor([2.], dtype=torch.float64)
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                rc = torch.tensor([invalid], dtype=torch.float64)
                with self.assertRaises(ValueError):
                    cone_terms(u, rp, u, rc, .1, .1)
                v = torch.tensor([[invalid, 0.]], dtype=torch.float64)
                with self.assertRaises(ValueError):
                    cone_terms(u, rp, v, rp, .1, .1)

    def test_auxiliary_has_no_distance_penalties(self):
        geo = geometry()
        # Equal radii and antipodal directions violate both removed guard thresholds.
        parent = torch.zeros(2, 8)
        parent[:, 0] = 1
        child = -parent
        for epoch in (0, 3, 8):
            with self.subTest(epoch=epoch):
                loss, metrics = geo.hnm_loss(child, parent, epoch)
                self.assertGreater(metrics["pair_distance"].item(), 1.5)
                torch.testing.assert_close(metrics["radius_eeg"], metrics["radius_material"])
                self.assertEqual(loss.item(), (metrics["cone_weight"]*metrics["cone"]).item())
                self.assertGreater(loss.item(), 0.)
                self.assertNotIn("gap", metrics)
                self.assertNotIn("far", metrics)
        self.assertNotIn("guard", config().train.hnm)

    def test_original_train_validation_targets_preserved_only_alignment_changes(self):
        for dataset in ("FACED", "FACED_def_c2", "SEED"):
            base_cfg, hnm_cfg = config("hyper_distance", dataset), config(dataset=dataset)
            for cfg in (base_cfg, hnm_cfg):
                # Exercise the existing frozen aligner path without real weights in a synthetic test.
                cfg.train.pretrained_text_proj = cfg.train.pretrained_image_proj = ""
            hnm_cfg.train.hnm.cone.weight = 0.
            torch.manual_seed(17)
            base = model(base_cfg)
            torch.manual_seed(17)
            added = model(hnm_cfg)
            base_parameters, added_parameters = dict(base.named_parameters()), dict(added.named_parameters())
            for key, parameter in base_parameters.items():
                torch.testing.assert_close(parameter, added_parameters[key], rtol=0, atol=0)
            self.assertIs(type(base), type(added))
            self.assertIs(type(added.distill_criterion), HNMPaperContrastiveLoss)
            self.assertNotIsInstance(base.distill_criterion, HNMPaperContrastiveLoss)
            eeg, text, image = torch.randn(6, 1024), torch.randn(6, 5, 1024), torch.randn(6, 1024)
            labels = torch.tensor([0, 1, 4, 5, 6, 8]) if dataset == "FACED_def_c2" else torch.tensor([0, 1, 2, 0, 1, 2])
            batch = (eeg, labels, text, image, torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 0, 0, 1, 1, 1]))
            for phase in ("training_step", "validation_step"):
                captured, values, alignments = [], [], []
                for module in (base, added):
                    module.train(phase == "training_step")
                    handle = module.distill_criterion.register_forward_pre_hook(
                        lambda _, args: captured.append(tuple(x.detach().clone() for x in args[:2])))
                    output_handle = module.distill_criterion.register_forward_hook(
                        lambda _, args, output: alignments.append(output.detach()))
                    torch.manual_seed(21)
                    values.append(getattr(module, phase)(batch, 0))
                    handle.remove()
                    output_handle.remove()
                for original, extended in zip(captured[0], captured[1]):
                    torch.testing.assert_close(original, extended, rtol=0, atol=0)
                self.assertEqual(captured[0][0].shape[0], 5 if dataset == "FACED_def_c2" else 6)
                torch.testing.assert_close(values[1]-values[0],
                                           hnm_cfg.train.w_clip*(alignments[1]-alignments[0]),
                                           atol=1e-5, rtol=1e-5)

    def test_auxiliary_hook_respects_original_total_loss_scaling(self):
        cfg = config()
        cfg.train.use_two_stage_projector = False
        cfg.train.w_clip = 2.5  # In-memory test only; stored original value remains 1.0.
        module = model(cfg)
        batch = (torch.randn(6, 1024), torch.tensor([0, 1, 2, 0, 1, 2]),
                 torch.randn(6, 5, 1024), torch.randn(6, 1024), torch.arange(6), torch.zeros(6))
        logs = {}
        module.log = lambda name, value, **kwargs: logs.update({name: value})
        torch.manual_seed(8)
        with_hnm = module.training_step(batch, 0)
        auxiliary = logs["ext/train/hnm_weighted"].clone()
        module.contrastive_geometry.hnm_loss = lambda eeg, targets, epoch: (eeg.sum()*0, {})
        torch.manual_seed(8)
        without = module.training_step(batch, 0)
        torch.testing.assert_close(with_hnm-without, cfg.train.w_clip*auxiliary, atol=3e-6, rtol=3e-5)

    def test_checkpoint_roundtrip_and_original_extraction_loading(self):
        cfg = config()
        cfg.train.use_two_stage_projector = False
        module = model(cfg)
        with torch.no_grad():
            module.distill_criterion.log_temperature.fill_(math.log(.04))
        state = module.state_dict()
        restored = model(cfg)
        restored.load_state_dict(state, strict=True)
        self.assertAlmostEqual(restored.distill_criterion.temperature.item(), .04)
        filtered = {key: value for key, value in state.items() if not key.startswith(("text_probe.", "image_probe."))}
        missing, unexpected = restored.load_state_dict(filtered, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(key.startswith(("text_probe.", "image_probe.")) for key in missing))
        changed = config()
        changed.train.use_two_stage_projector = False
        changed.train.hnm.cone.weight = .02
        with self.assertRaisesRegex(ValueError, "settings differ"):
            model(changed).load_state_dict(state, strict=True)
        changed = config()
        changed.train.use_two_stage_projector = False
        changed.train.hnm.contrastive.minimum_temperature = .02
        with self.assertRaisesRegex(ValueError, "settings differ"):
            model(changed).load_state_dict(state, strict=True)
        old = config()
        old.train.use_two_stage_projector = False
        old.train.hnm.spec_id = "hyperama_hnm_no_distance_guards_v3"
        with self.assertRaisesRegex(ValueError, "settings differ"):
            restored.load_state_dict(model(old).state_dict(), strict=False)

    def test_invalid_configuration_and_disabled_hnm(self):
        cfg = config()
        cfg.train.hnm.radius.initial = .8
        with self.assertRaisesRegex(ValueError, "initialization"):
            HNMGeometry(.1, cfg.train.hnm)
        cfg = config()
        cfg.train.use_two_stage_projector = False
        cfg.train.hnm.enabled = False
        original = model(cfg)
        self.assertIs(type(original.contrastive_geometry), ContrastiveGeometry)
        self.assertIs(type(original.distill_criterion), MultiModalInfoNCELoss)
        cfg.train.hnm.enabled = True
        cfg.train.contrastive.symmetric = False
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            model(cfg)
        for initial, minimum in ((.001, .01), (.07, 0.), (float("nan"), .01)):
            with self.subTest(initial=initial, minimum=minimum), self.assertRaises(ValueError):
                HNMPaperContrastiveLoss(geometry(), initial, minimum)

    def test_mixed_precision_input_has_finite_gradients(self):
        geo = geometry()
        x, y = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True), torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores = geo(x, y)
            criterion = HNMPaperContrastiveLoss(geo)
            loss = criterion(x, y)+geo.hnm_loss(x, y, 8)[0]
        self.assertEqual(scores.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertTrue(torch.isfinite(criterion.log_temperature.grad))

    def test_actual_faced_backbone_one_synthetic_backward(self):
        cfg = config()
        cfg.train.use_two_stage_projector = False
        cfg.model.use_ln_backbone = True  # Same setting applied by the existing FACED entrypoint.
        module = model(cfg, instantiate(cfg.model))
        batch = (torch.randn(6, 1, 32, 625), torch.tensor([0, 1, 2, 0, 1, 2]),
                 torch.randn(6, 5, 1024), torch.randn(6, 1024),
                 torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 0, 0, 1, 1, 1]))
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for head in (module.contrastive_geometry.eeg_radius, module.contrastive_geometry.material_radius):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.model.parameters() if p.grad is not None))
        optimizer = module.configure_optimizers()["optimizer"]
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertIn(id(module.contrastive_geometry.eeg_radius.weight), optimized)
        self.assertIn(id(module.distill_criterion.log_temperature), optimized)
        self.assertTrue(torch.isfinite(module.distill_criterion.log_temperature.grad))
        module.eval()
        with torch.no_grad():
            self.assertEqual(module(batch[0]).shape, (6, 1024))


if __name__ == "__main__":
    unittest.main()
