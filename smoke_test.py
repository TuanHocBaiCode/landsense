"""Offline synthetic integration tests. Run: python smoke_test.py (no dataset/weights download)."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

import landsense as ls


def main():
    torch.set_num_threads(2)
    ls.seed_all(7)
    a = np.zeros((64, 64, 13), dtype=np.uint16)
    a[..., 3], a[..., 7] = 100, 25
    assert np.isclose(ls.channels_from_ms(a, 'rgbndvi', ls.DEFAULT_BANDS)[0, 0, 3], -0.6)
    for mode, n in [('rgb', 3), ('rgbndvi', 4), ('ms', 13), ('ms10', 10)]:
        assert ls.channels_from_ms(a, mode, ls.DEFAULT_BANDS).shape[-1] == n
        cfg = ls.Config(mode=mode, num_workers=0, pretrained=False)
        model = ls.build_model(cfg)
        model.eval()
        x = torch.randn(2, n, 64, 64)
        with torch.no_grad():
            assert model(x).shape == (2, 10)
        if mode == 'rgb':
            assert sum(p.numel() for p in model.parameters()) == 391946
        cam, probs, cls = ls.gradcam(model, x[:1], model.features[-4])
        assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1.00001
        assert np.isclose(probs.sum(), 1)
    for strategy in ['frozen', 'finetune']:
        for mode, n in [('rgb', 3), ('rgbndvi', 4), ('ms', 13)]:
            cfg = ls.Config(model='resnet50', strategy=strategy, mode=mode,
                            pretrained=False, num_workers=0)
            model = ls.build_model(cfg)
            ls.train_mode(model)
            assert model.bn1.training == (strategy == 'finetune' and n != 3)
            assert model.layer4[0].bn1.training == (strategy == 'finetune')
            assert not model.layer1[0].bn1.training
            assert all(p.requires_grad for p in model.fc.parameters())
            if n > 3:
                assert torch.allclose(model.conv1.weight[:, 3], model.conv1.weight[:, :3].mean(1))
            model.eval()
            x = torch.randn(1, n, 64, 64)
            with torch.no_grad():
                assert model(x).shape == (1, 10)
            if strategy == 'frozen' and mode == 'rgb':
                cam, _, _ = ls.gradcam(model, x, model.layer4[-1])
                assert np.isfinite(cam).all()
            del model

    with tempfile.TemporaryDirectory(prefix='landsense_smoke_') as temp:
        root = Path(temp)
        rgb, ms = root / 'rgb', root / 'ms'
        rng = np.random.default_rng(123)
        for cls in ls.CLASSES:
            (rgb / cls).mkdir(parents=True)
            (ms / cls).mkdir(parents=True)
            for i in range(20):
                Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                    rgb / cls / f'{cls}_{i}.jpg')
                tifffile.imwrite(ms / cls / f'{cls}_{i}.tif',
                                 rng.integers(1, 5000, (64, 64, 13), dtype=np.uint16),
                                 photometric='minisblack')
        tifffile.imwrite(ms / 'SeaLake' / 'Jakarta_extra.tif', a, photometric='minisblack')
        parts, meta = ls.scan_dataset(rgb, ms, root / 'splits', strict_counts=False, full_audit=True)
        assert [len(parts[k]) for k in ['train', 'val', 'test']] == [140, 30, 30]
        assert len(meta['excluded_ms_only']) == 1
        parts2, meta2 = ls.scan_dataset(rgb, ms, root / 'splits', strict_counts=False, full_audit=False)
        assert meta['split_hash'] == meta2['split_hash']
        for name in parts:
            assert parts[name].sample_id.tolist() == parts2[name].sample_id.tolist()
        cfg = ls.Config(epochs=2, batch_size=32, num_workers=0, amp=False, pretrained=False)
        # Emulate a disconnect at start of epoch 2, then resume from last.pt.
        original = ls.run_epoch
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('simulated interruption')
            return original(*args, **kwargs)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with patch.object(ls, 'run_epoch', side_effect=interrupted):
                try:
                    ls.fit(cfg, parts, meta, rgb, ms, root / 'resumed')
                    raise AssertionError('Expected simulated interruption')
                except RuntimeError as exc:
                    assert str(exc) == 'simulated interruption'
            resumed = ls.fit(cfg, parts, meta, rgb, ms, root / 'resumed')
            direct = ls.fit(cfg, parts, meta, rgb, ms, root / 'direct')
            state_a, state_b = ls.load_checkpoint(resumed / 'last.pt'), ls.load_checkpoint(direct / 'last.pt')
            for key in state_a['model']:
                assert torch.equal(state_a['model'][key], state_b['model'][key]), key
            assert state_a['history'] == state_b['history']
            metrics = ls.evaluate_test(resumed, parts, meta, rgb, ms)
            assert metrics['n_test'] == 30 and 0 <= metrics['f1_macro'] <= 1
            ls.export_gradcam(resumed, parts['test'], rgb, ms, indices=[0, 1])
            assert (resumed / 'gradcam_examples.png').is_file()
            ls.summarize(root / 'resumed')
            assert ls.mcnemar(resumed, resumed)['p_exact'] == 1
        for fname in ['best.pt', 'last.pt', 'config.json', 'history.csv', 'test_predictions.csv',
                      'learning_curves.png', 'confusion_matrix.png', 'gradcam_examples.json']:
            assert (resumed / fname).is_file(), fname
    print('PASS: channels/NDVI, 4 CNN inputs, 6 ResNet variants, frozen BN, Grad-CAM,')
    print('pairing/splits, training, exact CPU resume, evaluation, summary, McNemar.')


if __name__ == '__main__':
    main()