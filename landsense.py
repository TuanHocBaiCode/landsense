"""LandSense: EuroSAT training utilities. Generated checkpoints are trusted local files."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
EXPECTED = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
# EuroSAT allBands order is a dataset-specific assumption, verified in the notebook.
DEFAULT_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                 'B09', 'B10', 'B11', 'B12', 'B8A']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding='utf-8')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def read_ms(path):
    a = tifffile.imread(path)
    if a.shape == (13, 64, 64):
        a = np.moveaxis(a, 0, -1)
    if a.shape != (64, 64, 13):
        raise ValueError(f'{path}: expected (64,64,13) or (13,64,64), got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError(f'{path}: non-finite pixels')
    return a.astype(np.float32)


def validate_bands(bands):
    if len(bands) != 13 or set(bands) != set(DEFAULT_BANDS):
        raise ValueError('BAND_ORDER must contain each of the 13 named Sentinel-2 bands once.')


def channels_from_ms(a, mode, bands):
    """All returned channels use RGB first; convert unsigned TIFF before subtraction."""
    validate_bands(bands)
    a = a.astype(np.float32, copy=False)
    rgb_idx = [bands.index(b) for b in ['B04', 'B03', 'B02']]
    if mode == 'rgb':
        return a[..., rgb_idx]
    if mode == 'rgbndvi':
        red, nir = a[..., bands.index('B04')], a[..., bands.index('B08')]
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.concatenate([a[..., rgb_idx], ndvi[..., None]], axis=-1)
    if mode == 'ms':
        order = rgb_idx + [i for i in range(13) if i not in rgb_idx]
        return a[..., order]
    if mode == 'ms10':
        keep = [i for i, b in enumerate(bands) if b not in ['B01', 'B09', 'B10']]
        order = rgb_idx + [i for i in keep if i not in rgb_idx]
        return a[..., order]
    raise ValueError(f'Unknown input: {mode}')


def scan_dataset(rgb_root, ms_root, out_dir, split_seed=42, strict_counts=True,
                 full_audit=True):
    """Pair by class/stem, ignore TIFF-only extras, persist one shared stratified split."""
    rgb_root, ms_root, out_dir = Path(rgb_root), Path(ms_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, extras = [], []
    for label, cls in enumerate(CLASSES):
        rgb_dir, ms_dir = rgb_root / cls, ms_root / cls
        if not rgb_dir.is_dir() or not ms_dir.is_dir():
            raise FileNotFoundError(f'Roots must directly contain all 10 class folders; missing {cls}.')
        def index(folder, extensions):
            found = {}
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in extensions:
                    if p.stem in found:
                        raise ValueError(f'Duplicate image stem: {p}')
                    found[p.stem] = p
            return found
        rgb = index(rgb_dir, {'.jpg', '.jpeg', '.png'})
        ms = index(ms_dir, {'.tif', '.tiff'})
        missing = sorted(set(rgb) - set(ms))
        if missing:
            raise ValueError(f'{cls}: {len(missing)} RGB images lack TIFF partners: {missing[:5]}')
        extras.extend(str(ms[k].relative_to(ms_root)) for k in sorted(set(ms) - set(rgb)))
        if strict_counts and len(rgb) != EXPECTED[label]:
            raise ValueError(f'{cls}: expected {EXPECTED[label]} RGB images, found {len(rgb)}')
        for stem in sorted(rgb):
            p, q = rgb[stem], ms[stem]
            rows.append(dict(sample_id=f'{cls}/{stem}', class_name=cls, label=label,
                             rgb=str(p.relative_to(rgb_root)), ms=str(q.relative_to(ms_root)),
                             rgb_bytes=p.stat().st_size, ms_bytes=q.stat().st_size))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No paired images found.')
    fingerprint = digest(df.to_dict('records'))
    meta = dict(fingerprint=fingerprint, split_seed=split_seed, classes=CLASSES,
                n_paired=len(df), excluded_ms_only=extras,
                rgb_root=str(rgb_root), ms_root=str(ms_root))
    # Check existing split identity before writing any files.
    meta_path = out_dir / 'dataset.json'
    split_paths = [out_dir / f'{name}.csv' for name in ['train', 'val', 'test']]
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if old['fingerprint'] != fingerprint or old['split_seed'] != split_seed:
            raise ValueError('Dataset/split changed. Choose a NEW RESULTS_DIR to preserve prior runs.')
    else:
        if any(p.exists() for p in split_paths):
            raise ValueError('Split files exist without dataset.json. Use a clean RESULTS_DIR.')
    if full_audit:
        band_sum = np.zeros(13, dtype=np.float64)
        band_pixels = 0
        for row in tqdm(df.itertuples(index=False), total=len(df), desc='Audit RGB + TIFF'):
            with Image.open(rgb_root / row.rgb) as im:
                im.load()
                if im.size != (64, 64):
                    raise ValueError(f'{row.rgb}: expected 64x64, got {im.size}')
            a = read_ms(ms_root / row.ms)
            band_sum += a.sum(axis=(0, 1), dtype=np.float64)
            band_pixels += a.shape[0] * a.shape[1]
        write_json(out_dir / 'band_audit.json', dict(
            means_in_file_order=(band_sum / band_pixels).tolist(),
            note='These means are descriptive audit only; never used for normalization.'))
    if not all(p.exists() for p in split_paths):
        if any(p.exists() for p in split_paths):
            raise ValueError('Incomplete saved split: use a new directory or restore missing files.')
        train, rest = train_test_split(df, test_size=0.30, stratify=df.label,
                                      random_state=split_seed)
        val, test = train_test_split(rest, test_size=0.50, stratify=rest.label,
                                    random_state=split_seed)
        for name, part in [('train', train), ('val', val), ('test', test)]:
            part.sort_values('sample_id').to_csv(out_dir / f'{name}.csv', index=False)
    write_json(meta_path, meta)
    parts = {n: pd.read_csv(out_dir / f'{n}.csv') for n in ['train', 'val', 'test']}
    identities = {n: set(p.sample_id) for n, p in parts.items()}
    if (identities['train'] & identities['val'] or identities['train'] & identities['test']
            or identities['val'] & identities['test']):
        raise ValueError('Overlapping split identities.')
    if set.union(*identities.values()) != set(df.sample_id):
        raise ValueError('Saved splits do not match scanned dataset.')
    expected_rows = df.set_index('sample_id')
    for part in parts.values():
        if part.sample_id.duplicated().any():
            raise ValueError('Duplicate rows in saved split.')
        for row in part.itertuples(index=False):
            ref = expected_rows.loc[row.sample_id]
            if (row.rgb, row.ms, row.label) != (ref.rgb, ref.ms, ref.label):
                raise ValueError(f'Tampered split row: {row.sample_id}')
    meta['split_hash'] = digest({n: p.to_dict('records') for n, p in parts.items()})
    print(f'Paired: {len(df):,}; excluded TIFF-only: {len(extras):,}')
    print(pd.DataFrame({n: p.class_name.value_counts() for n, p in parts.items()}).reindex(CLASSES))
    return parts, meta


def load_pixels(row, rgb_root, ms_root, mode, rgb_source, bands):
    if mode == 'rgb' and rgb_source == 'jpg':
        with Image.open(Path(rgb_root) / row.rgb) as im:
            return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    return channels_from_ms(read_ms(Path(ms_root) / row.ms), mode, bands)


def compute_stats(train, rgb_root, ms_root, mode, rgb_source, bands, cache_dir,
                  split_hash, model):
    # JPG ResNet follows ImageNet convention. TIFF-derived inputs use train-only moments.
    if mode == 'rgb' and rgb_source == 'jpg' and model == 'resnet50':
        return dict(mean=IMAGENET_MEAN, std=IMAGENET_STD, origin='ImageNet')
    key = digest(dict(split_hash=split_hash, mode=mode, rgb_source=rgb_source, bands=bands))
    dest = Path(cache_dir) / f'stats_{key[:16]}.json'
    if dest.exists():
        return json.loads(dest.read_text(encoding='utf-8'))
    total = square = None
    count = 0
    for row in tqdm(train.itertuples(index=False), total=len(train), desc=f'Train stats {mode}'):
        a = load_pixels(row, rgb_root, ms_root, mode, rgb_source, bands).astype(np.float64)
        flat = a.reshape(-1, a.shape[-1])
        if total is None:
            total = np.zeros(a.shape[-1])
            square = total.copy()
        total += flat.sum(axis=0)
        square += (flat * flat).sum(axis=0)
        count += len(flat)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 1e-12))
    result = dict(mean=mean.tolist(), std=std.tolist(), origin='train_only', key=key)
    write_json(dest, result)
    return result


class EuroSATDataset(Dataset):
    def __init__(self, frame, rgb_root, ms_root, cfg, stats, train=False):
        self.frame = frame.reset_index(drop=True)
        self.rgb_root, self.ms_root, self.cfg = rgb_root, ms_root, cfg
        self.mean = torch.tensor(stats['mean'])[:, None, None]
        self.std = torch.tensor(stats['std'])[:, None, None]
        self.train = train

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        a = load_pixels(row, self.rgb_root, self.ms_root, self.cfg.mode,
                        self.cfg.rgb_source, self.cfg.bands)
        x = torch.from_numpy(a.transpose(2, 0, 1).copy())
        if self.train:
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            if torch.rand(()) < 0.5:
                x = x.flip(-2)
            x = torch.rot90(x, int(torch.randint(0, 4, ()).item()), (-2, -1))
            # Joint geometry for every channel; no zero-fill borders from arbitrary rotations.
            # Omit color jitter in ALL modes to isolate the effect of extra channels.
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.cfg.image_size:
            x = TF.resize(x, [self.cfg.image_size] * 2, antialias=True,
                          interpolation=InterpolationMode.BILINEAR)
        return x, int(row.label)


@dataclass
class Config:
    model: str = 'cnn'
    strategy: str = 'scratch'
    mode: str = 'rgb'
    seed: int = 42
    split_seed: int = 42
    rgb_source: str = 'tif'
    bands: tuple = tuple(DEFAULT_BANDS)
    epochs: int = 40
    batch_size: int = 32
    patience: int = 7
    lr: float | None = None
    weight_decay: float = 1e-4
    num_workers: int = 2
    amp: bool = True
    pretrained: bool = True

    @property
    def image_size(self):
        return 64 if self.model == 'cnn' else 224

    @property
    def channels(self):
        return {'rgb': 3, 'rgbndvi': 4, 'ms': 13, 'ms10': 10}[self.mode]

    @property
    def learning_rate(self):
        return self.lr if self.lr is not None else (1e-4 if self.strategy == 'finetune' else 1e-3)

    def validate(self):
        validate_bands(self.bands)
        if self.model not in ['cnn', 'resnet50'] or self.mode not in ['rgb', 'rgbndvi', 'ms', 'ms10']:
            raise ValueError('Unsupported model/input.')
        if (self.model == 'cnn' and self.strategy != 'scratch') or (
                self.model == 'resnet50' and self.strategy not in ['frozen', 'finetune']):
            raise ValueError('CNN uses scratch; ResNet50 uses frozen or finetune.')
        if self.rgb_source not in ['jpg', 'tif']:
            raise ValueError('rgb_source must be jpg or tif.')
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError('epochs, batch_size, patience must be positive.')


class SmallCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        blocks = []
        for out in [32, 64, 128, 256]:
            blocks.extend([nn.Conv2d(channels, out, 3, padding=1), nn.BatchNorm2d(out),
                           nn.ReLU(), nn.MaxPool2d(2)])
            channels = out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, len(CLASSES)))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def build_model(cfg, download=True):
    cfg.validate()
    if cfg.model == 'cnn':
        return SmallCNN(cfg.channels)
    weights = ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained and download else None
    model = resnet50(weights=weights)
    if cfg.channels != 3:
        old = model.conv1.weight.detach().clone()
        model.conv1 = nn.Conv2d(cfg.channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old
            model.conv1.weight[:, 3:] = old.mean(1, keepdim=True).expand(-1, cfg.channels - 3, -1, -1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if cfg.strategy == 'finetune':
        for p in model.layer4.parameters():
            p.requires_grad = True
        if cfg.channels != 3:
            for module in [model.conv1, model.bn1]:
                for p in module.parameters():
                    p.requires_grad = True
    # frozen means every backbone weight, including inflated conv1, is frozen.
    return model


def train_mode(model):
    model.train()
    # model.train() would otherwise update running means in frozen BatchNorm layers.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None)


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    train_mode(model) if training else model.eval()
    ys, preds, total_loss = [], [], 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc='train' if training else 'validation'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(x).all():
                raise FloatingPointError('Input contains NaN/Inf after preprocessing; check image and train statistics.')
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                precision = 'AMP/FP16' if amp else 'FP32'
                raise FloatingPointError(
                    f'Non-finite model output/loss in {precision}; '
                    f'input range=[{x.min().item():.4g}, {x.max().item():.4g}]. '
                    'For AMP, set RESNET_USE_AMP=False and recreate configs. '
                    'If FP32 also fails, inspect checkpoint weights and CUDA/cuDNN.')
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            ys.extend(y.detach().cpu().tolist())
            preds.extend(logits.detach().argmax(1).cpu().tolist())
    p, r, f, _ = precision_recall_fscore_support(ys, preds, average='macro',
                                               labels=list(range(10)), zero_division=0)
    return dict(loss=total_loss / len(ys), accuracy=accuracy_score(ys, preds),
                precision_macro=p, recall_macro=r, f1_macro=f)


def save_checkpoint(path, value):
    tmp = Path(path).with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # Only load checkpoints produced by this project, never untrusted downloaded .pt files.
    return torch.load(path, map_location='cpu', weights_only=False)


# Completed CNN runs from the previous source are numerically compatible: this
# revision changes ResNet precision and adds diagnostics; CNN updates are unchanged.
COMPATIBLE_CNN_CODE = {'dcae5063fa8342828fba81f9eef955532b70b00effbaee7d272d508ac23d4933'}


def find_compatible_cnn_run(cfg, meta, results_dir):
    if cfg.model != 'cnn':
        return None
    expected = json.loads(json.dumps(asdict(cfg)))
    candidates = []
    for path in (Path(results_dir) / 'runs').glob('cnn_*/config.json'):
        parent = path.parent
        if not all((parent / name).is_file() for name in ['done.json', 'best.pt', 'history.csv']):
            continue
        info = json.loads(path.read_text(encoding='utf-8'))
        if (info.get('code_sha256') in COMPATIBLE_CNN_CODE
                and info.get('config') == expected
                and info.get('split_hash') == meta['split_hash']):
            candidates.append(parent)
    if len(candidates) > 1:
        raise ValueError('Multiple compatible completed CNN runs: inspect run directories.')
    return candidates[0] if candidates else None


def fit(cfg, parts, meta, rgb_root, ms_root, results_dir):
    cfg.validate()
    compatible = find_compatible_cnn_run(cfg, meta, results_dir)
    if compatible is not None:
        print(f'Reusing completed compatible CNN run: {compatible.name}')
        return compatible
    seed_all(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (device.type == 'cuda' and cfg.model == 'resnet50' and cfg.amp
            and 'GTX 1650' in torch.cuda.get_device_name(0)):
        from dataclasses import replace
        cfg = replace(cfg, amp=False)
        print('GTX 1650: using FP32 for ResNet50 to avoid the observed FP16 non-finite loss.')
    if device.type == 'cpu':
        print('CPU runtime: check the NVIDIA driver and the CUDA-enabled PyTorch installation.')
    identity = dict(config=asdict(cfg), split_hash=meta['split_hash'],
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run_name = f'{cfg.model}_{cfg.strategy}_{cfg.mode}_s{cfg.seed}_{digest(identity)[:8]}'
    run_dir = Path(results_dir) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'landsense_used.py').write_bytes(Path(__file__).read_bytes())
    if (run_dir / 'done.json').exists():
        print(f'Already trained: {run_name}')
        return run_dir
    stats = compute_stats(parts['train'], rgb_root, ms_root, cfg.mode, cfg.rgb_source,
                          cfg.bands, Path(results_dir) / 'stats', meta['split_hash'], cfg.model)
    write_json(run_dir / 'config.json', dict(**identity, stats=stats, classes=CLASSES,
                                          environment=environment()))
    datasets = {n: EuroSATDataset(parts[n], rgb_root, ms_root, cfg, stats, train=n == 'train')
                for n in ['train', 'val']}
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {n: DataLoader(ds, batch_size=cfg.batch_size, shuffle=n == 'train',
                            num_workers=cfg.num_workers, pin_memory=device.type == 'cuda',
                            worker_init_fn=seed_worker, generator=generator if n == 'train' else None,
                            persistent_workers=False) for n, ds in datasets.items()}
    # Do not download ImageNet weights on resume: saved model contains all backbone weights.
    last = run_dir / 'last.pt'
    model = build_model(cfg, download=not last.exists()).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    amp = cfg.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best, stale, history, elapsed = 0, -1.0, 0, [], 0.0
    if last.exists():
        state = load_checkpoint(last)
        if state['identity'] != json.loads(json.dumps(identity)):
            raise ValueError('Checkpoint config/split mismatch.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        # Optimizer state tensors follow their parameter device after load_state_dict.
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start, best, stale = state['epoch'] + 1, state['best_f1'], state['stale']
        history, elapsed = state['history'], state['elapsed_seconds']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        generator.set_state(state['rng_loader'])
        if device.type == 'cuda' and state['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        print(f'Resuming epoch {start + 1}, best validation F1={best:.4f}')
    for epoch in range(start, cfg.epochs):
        if stale >= cfg.patience:
            break
        t0 = time.perf_counter()
        tr = run_epoch(model, loaders['train'], device, optimizer, scaler, amp)
        va = run_epoch(model, loaders['val'], device, amp=amp)
        scheduler.step(va['f1_macro'])
        elapsed += time.perf_counter() - t0
        record = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]['lr'],
                      **{f'train_{k}': v for k, v in tr.items()},
                      **{f'val_{k}': v for k, v in va.items()})
        history.append(record)
        improved = va['f1_macro'] > best
        if improved:
            best, stale = va['f1_macro'], 0
        else:
            stale += 1
        # Save CPU tensors: checkpoints are portable and do not duplicate GPU state on load.
        state = dict(identity=json.loads(json.dumps(identity)),
                     model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     epoch=epoch, best_f1=best, stale=stale, history=history,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), elapsed_seconds=elapsed,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(), rng_loader=generator.get_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        if improved:
            save_checkpoint(run_dir / 'best.pt', dict(model=state['model'], epoch=epoch,
                                                      validation=va, identity=state['identity']))
        save_checkpoint(last, state)
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        print(f'{run_name} | {epoch + 1}/{cfg.epochs} | val F1 {va["f1_macro"]:.4f} | best {best:.4f}')
    summary = dict(best_val_f1=best, epochs_completed=len(history),
                   train_seconds=elapsed, device=str(device),
                   params_total=sum(p.numel() for p in model.parameters()),
                   params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    plot_history(run_dir)
    write_json(run_dir / 'done.json', summary)
    return run_dir


def restore_model(run_dir, device=None):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    cfg = Config(**info['config'])
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, download=False)
    model.load_state_dict(load_checkpoint(run_dir / 'best.pt')['model'])
    return model.to(device).eval(), cfg, info, device


def evaluate_test(run_dir, parts, meta, rgb_root, ms_root):
    """Call only after experiment plan is fixed. No test scores are used by fit()."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    if info['split_hash'] != meta['split_hash']:
        raise ValueError('Test split does not match training split.')
    if (run_dir / 'test_metrics.json').exists():
        return json.loads((run_dir / 'test_metrics.json').read_text(encoding='utf-8'))
    model, cfg, info, device = restore_model(run_dir)
    ds = EuroSATDataset(parts['test'], rgb_root, ms_root, cfg, info['stats'])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=device.type == 'cuda')
    probabilities, ys = [], []
    # Warm up separately; forward timing excludes data loading/transfer and Grad-CAM.
    x0, _ = ds[0]
    with torch.no_grad():
        for _ in range(3):
            model(x0[None].to(device))
    forward_seconds = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Final test'):
            x = x.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - t0
            probabilities.append(logits.softmax(1).cpu().numpy())
            ys.extend(y.tolist())
    probs = np.concatenate(probabilities)
    preds = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(ys, preds, labels=range(10),
                                               average='macro', zero_division=0)
    metrics = dict(accuracy=accuracy_score(ys, preds), precision_macro=p, recall_macro=r,
                   f1_macro=f, forward_ms_per_image=1000 * forward_seconds / len(ds),
                   inference_batch_size=cfg.batch_size, inference_device=str(device),
                   inference_precision='float32', n_test=len(ds),
                   **json.loads((run_dir / 'done.json').read_text(encoding='utf-8')))
    pred_frame = parts['test'][['sample_id', 'class_name', 'label']].reset_index(drop=True).copy()
    pred_frame['prediction'] = preds
    pred_frame['confidence'] = probs.max(1)
    for i, cls in enumerate(CLASSES):
        pred_frame[f'p_{cls}'] = probs[:, i]
    pred_frame.to_csv(run_dir / 'test_predictions.csv', index=False)
    report = classification_report(ys, preds, labels=list(range(10)), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(run_dir / 'classification_report.csv')
    cm = confusion_matrix(ys, preds, labels=list(range(10)))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(run_dir / 'confusion_matrix.csv')
    plot_confusion(cm, run_dir / 'confusion_matrix.png')
    write_json(run_dir / 'test_metrics.json', metrics)
    return metrics


def plot_history(run_dir):
    import matplotlib.pyplot as plt
    df = pd.read_csv(Path(run_dir) / 'history.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ['loss', 'accuracy', 'f1_macro']):
        for split in ['train', 'val']:
            ax.plot(df.epoch, df[f'{split}_{key}'], label=split)
        ax.set(xlabel='Epoch', ylabel=key)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / 'learning_curves.png', dpi=160)
    plt.close(fig)


def plot_confusion(cm, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES,
                yticklabels=CLASSES, ax=ax)
    ax.set(xlabel='Predicted label', ylabel='True label')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gradcam(model, x, layer, class_index=None):
    """Native Grad-CAM; also works when every backbone parameter is frozen."""
    activations = []
    hook = layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
    model.eval()
    try:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            logits = model(x)
            cls = int(logits.argmax(1).item()) if class_index is None else class_index
            a = activations[0]
            gradient = torch.autograd.grad(logits[0, cls], a)[0]
            cam = (gradient.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1).detach().cpu().numpy()[0], cls
    finally:
        hook.remove()


def export_gradcam(run_dir, test_frame, rgb_root, ms_root, indices=None):
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    model, cfg, info, device = restore_model(run_dir)
    ds = EuroSATDataset(test_frame, rgb_root, ms_root, cfg, info['stats'])
    pred = pd.read_csv(run_dir / 'test_predictions.csv')
    if indices is None:
        correct = pred[pred.label == pred.prediction].sort_values('confidence', ascending=False)
        wrong = pred[pred.label != pred.prediction].sort_values('confidence', ascending=False)
        pairs = [{0, 6}, {2, 5}, {3, 8}]
        confused = wrong.loc[[{int(r.label), int(r.prediction)} in pairs
                          for r in wrong.itertuples()]]
        indices = list(dict.fromkeys(list(correct.head(3).index) +
                                     list(confused.head(3).index) + list(wrong.head(3).index)))
    if not indices:
        return
    layer = model.features[-4] if cfg.model == 'cnn' else model.layer4[-1]
    fig, axes = plt.subplots(len(indices), 2, figsize=(9, 4 * len(indices)), squeeze=False)
    selections = []
    for row_idx, idx in enumerate(indices):
        x, y = ds[idx]
        cam, probs, cls = gradcam(model, x[None].to(device), layer)
        row = test_frame.iloc[idx]
        # RGB visualization comes from the same input source; percentile stretch is display only.
        rgb = load_pixels(row, rgb_root, ms_root, 'rgb', cfg.rgb_source, cfg.bands)
        lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        cam64 = F.interpolate(torch.tensor(cam)[None, None], (64, 64),
                              mode='bilinear', align_corners=False)[0, 0].numpy()
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f'True: {CLASSES[y]}')
        axes[row_idx, 1].imshow(rgb)
        axes[row_idx, 1].imshow(cam64, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'Pred: {CLASSES[cls]} ({probs[cls]:.1%})')
        for ax in axes[row_idx]:
            ax.axis('off')
        selections.append(dict(sample_id=row.sample_id, true=CLASSES[y], predicted=CLASSES[cls],
                               confidence=float(probs[cls]), cam_class=CLASSES[cls],
                               heatmap_nonzero=bool(cam.max() > 0)))
    fig.tight_layout()
    fig.savefig(run_dir / 'gradcam_examples.png', dpi=150)
    plt.close(fig)
    write_json(run_dir / 'gradcam_examples.json', selections)


def summarize(results_dir):
    rows = []
    for path in sorted((Path(results_dir) / 'runs').glob('*/test_metrics.json')):
        info = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        cfg = info['config']
        protocol_code = info['code_sha256']
        current_code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if cfg['model'] == 'cnn' and protocol_code in COMPATIBLE_CNN_CODE | {current_code}:
            protocol_code = 'cnn-training-v1-compatible'
        rows.append(dict(run=path.parent.name, model=cfg['model'], strategy=cfg['strategy'],
                         mode=cfg['mode'], seed=cfg['seed'], rgb_source=cfg['rgb_source'],
                         protocol=digest(dict(config={k: v for k, v in cfg.items() if k != 'seed'},
                                              code_sha256=protocol_code)),
                         split_hash=info['split_hash'],
                         **json.loads(path.read_text(encoding='utf-8'))))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No test evaluations yet.')
    df.to_csv(Path(results_dir) / 'all_runs.csv', index=False)
    if df.duplicated(['protocol', 'split_hash', 'seed']).any():
        raise ValueError('Duplicate seed within the same protocol; inspect run directories.')
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
               'train_seconds', 'forward_ms_per_image', 'params_trainable']
    group = ['model', 'strategy', 'mode', 'rgb_source', 'protocol', 'split_hash']
    agg = df.groupby(group, dropna=False)[metrics].agg(['mean', 'std', 'count'])
    agg.to_csv(Path(results_dir) / 'summary_mean_std.csv')
    return df, agg


def mcnemar(run_a, run_b):
    from scipy.stats import binomtest
    a = pd.read_csv(Path(run_a) / 'test_predictions.csv')
    b = pd.read_csv(Path(run_b) / 'test_predictions.csv')
    merged = a[['sample_id', 'label', 'prediction']].merge(
        b[['sample_id', 'label', 'prediction']], on='sample_id', suffixes=('_a', '_b'),
        validate='one_to_one')
    if len(merged) != len(a) or len(a) != len(b) or not (merged.label_a == merged.label_b).all():
        raise ValueError('McNemar requires the exact same labeled test images.')
    ca, cb = merged.prediction_a == merged.label_a, merged.prediction_b == merged.label_b
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(a_correct_b_wrong=a_only, a_wrong_b_correct=b_only,
                p_exact=binomtest(a_only, a_only + b_only, 0.5).pvalue
                if a_only + b_only else 1.0)
