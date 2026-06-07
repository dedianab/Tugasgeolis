# ============================================================
# ERT + IP Inversion Pipeline — Lintasan 6
# File input : YANG_BARU_YANG_BENER.dat
# Array      : Wenner-Schlumberger (kode 7)
# Spasi dasar: 3.0 m
# Datum      : 235 pengukuran
# Data       : ERT + IP (Chargeability, msec)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. IMPORT LIBRARY
# ============================================================
try:
    import pygimli as pg
    from pygimli.physics import ert
    print(f"pyGIMLi version : {pg.__version__}")
except ImportError:
    print("pyGIMLi tidak ditemukan. Install: pip install pygimli")
    raise

print("Import selesai.")

# ============================================================
# 1. FUNGSI-FUNGSI UTAMA
# ============================================================

ARRAY_CODES = {
    1: 'Wenner-Alpha',
    2: 'Pole-Pole',
    3: 'Dipole-Dipole',
    5: 'Pole-Dipole',
    7: 'Wenner-Schlumberger',
    8: 'Gradient'
}

# ----------------------------------------------------------
# 1.1 Parser Format Res2Dinv
# ----------------------------------------------------------
def parse_res2dinv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    title          = lines[0].strip()
    unit_spacing   = float(lines[1].strip())
    array_code     = int(lines[2].strip())
    n_data         = int(lines[3].strip())
    pos_convention = int(lines[4].strip())
    ip_flag        = int(lines[5].strip())

    meta = {
        'title'         : title,
        'unit_spacing'  : unit_spacing,
        'array_code'    : array_code,
        'array_name'    : ARRAY_CODES.get(array_code, f'Unknown({array_code})'),
        'n_data'        : n_data,
        'pos_convention': pos_convention,
        'ip_flag'       : ip_flag,
        'ip_name'       : None,
        'ip_unit'       : None,
        'ip_delay'      : None,
        'ip_integration': None,
    }

    data_start = 6
    if ip_flag == 1:
        meta['ip_name'] = lines[6].strip()
        meta['ip_unit'] = lines[7].strip()
        line8 = lines[8].strip()
        if ',' in line8:
            parts = line8.split(',')
            meta['ip_delay']       = float(parts[0])
            meta['ip_integration'] = float(parts[1])
            data_start = 9
        else:
            meta['ip_delay']       = float(line8)
            meta['ip_integration'] = float(lines[9].strip())
            data_start = 10

    data = []
    for line in lines[data_start:]:
        vals = line.strip().replace(',', ' ').split()
        if not vals or vals[0] == '0':
            break
        try:
            row = [float(v) for v in vals]
            if len(row) >= 4:
                data.append(row)
        except ValueError:
            continue

    return meta, np.array(data)


def print_summary(meta, data):
    has_ip = meta['ip_flag'] == 1 and data.shape[1] >= 5
    x, a, n, rho = data[:,0], data[:,1], data[:,2], data[:,3]
    print("=" * 62)
    print(f"  Lintasan      : {meta['title']}")
    print(f"  Array         : {meta['array_name']} (kode {meta['array_code']})")
    print(f"  Spasi dasar   : {meta['unit_spacing']} m")
    print(f"  N datum       : {meta['n_data']} (parsed: {len(data)})")
    print(f"  Konvensi x    : {'Midpoint (1)' if meta['pos_convention']==1 else 'First electrode (0)'}")
    print(f"  Data IP       : {'Ya' if meta['ip_flag']==1 else 'Tidak'}")
    if meta['ip_flag'] == 1:
        print(f"    Besaran     : {meta['ip_name']} ({meta['ip_unit']})")
        print(f"    Delay       : {meta['ip_delay']} s")
        print(f"    Integration : {meta['ip_integration']} s")
    print("-" * 62)
    print(f"  x range       : {x.min():.1f} - {x.max():.1f} m")
    print(f"  a values      : {sorted(np.unique(a).tolist())}")
    print(f"  n range       : {int(n.min())} - {int(n.max())}")
    print(f"  rhoa range    : {rho.min():.3f} - {rho.max():.3f} ohm.m")
    if has_ip:
        ip = data[:, 4]
        print(f"  IP range      : {ip.min():.1f} - {ip.max():.1f} {meta['ip_unit']}")
    print("=" * 62)


# ----------------------------------------------------------
# 1.2 Geometri Elektroda
# ----------------------------------------------------------
def compute_amnb(x, a, n, array_code, pos_convention):
    """
    Wenner-Schlumberger (kode 7), first electrode (pos_convention=0):
      A = x
      M = x + n*a
      N = x + (n+1)*a
      B = x + (2n+1)*a
    """
    if pos_convention == 1:
        if array_code == 7:
            A = x - (2*n + 1)*a / 2
            M = x - a / 2
            N = x + a / 2
            B = x + (2*n + 1)*a / 2
        elif array_code == 3:
            A = x - (n + 2)*a / 2
            B = A + a
            M = A + (n + 1)*a
            N = A + (n + 2)*a
        else:
            raise ValueError(f"Array {array_code} tidak didukung (midpoint)")
    else:
        if array_code == 7:
            A = x
            M = x + n*a
            N = x + (n + 1)*a
            B = x + (2*n + 1)*a
        elif array_code == 3:
            A = x
            B = x + a
            M = x + (n + 1)*a
            N = x + (n + 2)*a
        else:
            raise ValueError(f"Array {array_code} tidak didukung (first electrode)")
    return A, M, N, B


def geometric_factor(n, a, array_code):
    if array_code == 7:
        return np.pi * n * (n + 1) * a
    elif array_code == 3:
        return np.pi * n * (n + 1) * (n + 2) * a
    else:
        raise ValueError(f"Geometric factor untuk array {array_code} tidak diketahui")


def pseudo_depth(n, a, array_code):
    if array_code == 7:
        return n * a / 2.0
    elif array_code == 3:
        return n * a * 0.195
    else:
        return n * a / 2.0


# ----------------------------------------------------------
# 1.3 Build DataContainerERT
# ----------------------------------------------------------
def build_ert_container(meta, data, include_ip=False):
    x, a_col, n_col, rho = data[:,0], data[:,1], data[:,2], data[:,3]
    A, M, N, B = compute_amnb(x, a_col, n_col, meta['array_code'], meta['pos_convention'])
    K = geometric_factor(n_col, a_col, meta['array_code'])

    all_pos = np.unique(np.round(np.concatenate([A, M, N, B]), 4))
    scheme = pg.DataContainerERT()
    for xe in all_pos:
        scheme.createSensor(pg.Pos(xe, 0.0))

    pos_to_idx = {round(xe, 4): i for i, xe in enumerate(all_pos)}
    a_idx = np.array([pos_to_idx[round(A[i],4)] for i in range(len(rho))], dtype=int)
    b_idx = np.array([pos_to_idx[round(B[i],4)] for i in range(len(rho))], dtype=int)
    m_idx = np.array([pos_to_idx[round(M[i],4)] for i in range(len(rho))], dtype=int)
    n_idx = np.array([pos_to_idx[round(N[i],4)] for i in range(len(rho))], dtype=int)

    scheme.resize(len(rho))
    scheme.set('a', pg.IVector(a_idx.tolist()))
    scheme.set('b', pg.IVector(b_idx.tolist()))
    scheme.set('m', pg.IVector(m_idx.tolist()))
    scheme.set('n', pg.IVector(n_idx.tolist()))
    scheme.set('k',    pg.Vector(K.tolist()))
    scheme.set('rhoa', pg.Vector(rho.tolist()))
    scheme.set('err',  pg.Vector(np.full(len(rho), 0.05).tolist()))

    if include_ip and meta['ip_flag'] == 1 and data.shape[1] >= 5:
        scheme.set('ip', pg.Vector(data[:,4].tolist()))

    scheme.markValid(scheme['rhoa'] > 0)
    return scheme


# ----------------------------------------------------------
# 1.4 Plot Pseudosection
# ----------------------------------------------------------
def plot_pseudosection(x, z, values, ax, title, cbar_label,
                       cmap='Spectral_r', log_scale=True, n_levels=50):
    xi = np.linspace(x.min(), x.max(), 350)
    zi = np.linspace(z.min(), z.max(), 100)
    XI, ZI = np.meshgrid(xi, zi)
    VI = griddata((x, z), values, (XI, ZI), method='linear')

    if log_scale:
        norm = mcolors.LogNorm(vmin=np.nanmin(values[values>0]), vmax=np.nanmax(values))
    else:
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    cf = ax.contourf(XI, ZI, VI, levels=n_levels, cmap=cmap, norm=norm, alpha=0.8)
    ax.scatter(x, z, c=values, cmap=cmap, norm=norm, s=15,
               marker='v', edgecolors='none', zorder=3, alpha=0.7)
    plt.colorbar(cf, ax=ax, label=cbar_label, shrink=0.85, pad=0.02)
    ax.invert_yaxis()
    ax.set_ylabel('Pseudo-depth (m)', fontsize=9)
    ax.set_xlim(x.min()-2, x.max()+2)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')


# ============================================================
# 2. LOAD DATA
# ============================================================
DATA_FILE = "YANG_BARU_YANG_BENER.dat"

meta, raw = parse_res2dinv(DATA_FILE)
print_summary(meta, raw)

x_pos = raw[:, 0]
a_col = raw[:, 1]
n_col = raw[:, 2]
rhoa  = raw[:, 3]

if meta['ip_flag'] == 1 and raw.shape[1] >= 5:
    ip_val = raw[:, 4]
    print(f"\nKolom IP tersedia: {len(ip_val)} nilai, satuan {meta['ip_unit']}")
else:
    ip_val = None
    print("\nTidak ada kolom IP.")

z_pse = pseudo_depth(n_col, a_col, meta['array_code'])
print(f"\nJumlah elektroda unik: {len(np.unique(np.round(x_pos, 4)))}")


# ============================================================
# 2.5 QUALITY CONTROL (QC) & DATA FILTERING
# ============================================================
def qc_ert_ip(data, meta,
              remove_negative_rhoa=True,
              use_modified_zscore=False,
              zscore_threshold=7.0,
              use_iqr_per_level=False,
              iqr_multiplier=6.0,
              remove_negative_ip=False,
              ip_zscore_threshold=5.0,
              verbose=True):

    n      = len(data)
    rhoa_d = data[:, 3]
    n_col_d= data[:, 2]
    has_ip = meta['ip_flag'] == 1 and data.shape[1] >= 5
    ip_d   = data[:, 4] if has_ip else None

    mask   = np.ones(n, dtype=bool)
    report = {'total': n, 'removed_by': {}, 'removed_idx': []}

    def _flag(condition, label):
        bad   = mask & condition
        count = int(bad.sum())
        if count > 0:
            report['removed_by'][label] = count
            report['removed_idx'].extend(np.where(bad)[0].tolist())
            if verbose:
                print(f"  [{len(report['removed_by']):02d}] {label:<44s}: {count:>4d} datum")
        mask[:] &= ~bad

    if verbose:
        print("\nQC Report")
        print("-" * 62)

    if remove_negative_rhoa:
        _flag(rhoa_d <= 0, "rhoa <= 0 (invalid)")

    if use_modified_zscore:
        vals = rhoa_d[mask]
        med  = np.median(vals)
        mad  = np.median(np.abs(vals - med))
        if mad < 1e-10:
            mad = np.std(vals) * 0.6745 + 1e-10
        mz = 0.6745 * np.abs(rhoa_d - med) / mad
        _flag(mask & (mz > zscore_threshold), f"rhoa Modified Z-score > {zscore_threshold}")

    if use_iqr_per_level:
        bad_iqr = np.zeros(n, dtype=bool)
        for nv in np.unique(n_col_d):
            idx = np.where((n_col_d == nv) & mask)[0]
            if len(idx) < 5:
                continue
            q1, q3 = np.percentile(rhoa_d[idx], [25, 75])
            iqr = q3 - q1
            bad_in_level = (rhoa_d[idx] < q1 - iqr_multiplier*iqr) | \
                           (rhoa_d[idx] > q3 + iqr_multiplier*iqr)
            bad_iqr[idx[bad_in_level]] = True
        _flag(mask & bad_iqr, f"rhoa IQR x{iqr_multiplier} per n-level")

    if has_ip and remove_negative_ip:
        _flag(mask & (ip_d < 0), "IP negatif (artefak)")

    if has_ip:
        vals_ip = ip_d[mask]
        if len(vals_ip) > 4:
            med_ip = np.median(vals_ip)
            mad_ip = np.median(np.abs(vals_ip - med_ip))
            if mad_ip < 1e-10:
                mad_ip = np.std(vals_ip) * 0.6745 + 1e-10
            mz_ip = 0.6745 * np.abs(ip_d - med_ip) / mad_ip
            _flag(mask & (mz_ip > ip_zscore_threshold),
                  f"IP Modified Z-score > {ip_zscore_threshold}")

    report['n_valid']     = int(mask.sum())
    report['n_removed']   = n - report['n_valid']
    report['pct_removed'] = report['n_removed'] / n * 100

    if verbose:
        print("-" * 62)
        print(f"  Total dihapus   : {report['n_removed']:>4d} datum ({report['pct_removed']:.1f}%)")
        print(f"  Data valid      : {report['n_valid']:>4d} datum")

    return mask, report


qc_mask, qc_report = qc_ert_ip(
    raw, meta,
    remove_negative_rhoa = True,
    use_modified_zscore  = True,
    zscore_threshold     = 7.0,
    use_iqr_per_level    = True,
    iqr_multiplier       = 6.0,
    remove_negative_ip   = True,
    ip_zscore_threshold  = 5.0,
    verbose              = True
)

data_clean = raw[qc_mask]
x_pos = data_clean[:, 0]
a_col = data_clean[:, 1]
n_col = data_clean[:, 2]
rhoa  = data_clean[:, 3]
z_pse = pseudo_depth(n_col, a_col, meta['array_code'])

if meta['ip_flag'] == 1 and data_clean.shape[1] >= 5:
    ip_val = data_clean[:, 4]
else:
    ip_val = None

print(f"\nVariabel kerja diperbarui -> {len(data_clean)} datum bersih.")


# -- Plot QC distribusi --
fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor='white')
fig.suptitle(f'QC - Distribusi Data Sebelum vs Sesudah | {meta["title"]}',
             fontsize=11, fontweight='bold')

rhoa_raw   = raw[:, 3]
rhoa_clean = data_clean[:, 3]

axes[0].hist(rhoa_raw,   bins=50, alpha=0.55, color='#e74c3c',
             label=f'Raw  (N={len(rhoa_raw)})', edgecolor='none')
axes[0].hist(rhoa_clean, bins=50, alpha=0.65, color='#2ecc71',
             label=f'Clean (N={len(rhoa_clean)})', edgecolor='none')
axes[0].set_xlabel('Apparent Resistivity (ohm.m)', fontsize=9)
axes[0].set_ylabel('Frekuensi', fontsize=9)
axes[0].set_title('Distribusi rho_a', fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, linestyle='--')

n_vals         = sorted(np.unique(raw[:, 2]).astype(int).tolist())
data_box_raw   = [rhoa_raw[raw[:, 2] == nv]          for nv in n_vals]
data_box_clean = [rhoa_clean[data_clean[:, 2] == nv] for nv in n_vals]

pos_raw   = [i - 0.18 for i in range(len(n_vals))]
pos_clean = [i + 0.18 for i in range(len(n_vals))]

bp1 = axes[1].boxplot(data_box_raw, positions=pos_raw, widths=0.32, patch_artist=True,
                       boxprops=dict(facecolor='#f1948a', alpha=0.7),
                       medianprops=dict(color='#c0392b', linewidth=1.5),
                       flierprops=dict(marker='x', markersize=3, markeredgecolor='#e74c3c'))
bp2 = axes[1].boxplot(data_box_clean, positions=pos_clean, widths=0.32, patch_artist=True,
                       boxprops=dict(facecolor='#a9dfbf', alpha=0.7),
                       medianprops=dict(color='#1e8449', linewidth=1.5),
                       flierprops=dict(marker='x', markersize=3, markeredgecolor='#27ae60'))

axes[1].set_xticks(range(len(n_vals)))
axes[1].set_xticklabels([f'n={nv}' for nv in n_vals], fontsize=7)
axes[1].set_xlabel('Level investigasi', fontsize=9)
axes[1].set_ylabel('rho_a (ohm.m)', fontsize=9)
axes[1].set_title('Boxplot per n-level', fontsize=10)
axes[1].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Raw', 'Clean'], fontsize=8)
axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('QC_distribusi_rhoa.png', dpi=150, bbox_inches='tight')
plt.show()

if meta['ip_flag'] == 1 and raw.shape[1] >= 5:
    ip_raw   = raw[:, 4]
    ip_clean = data_clean[:, 4]
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='white')
    ax.hist(ip_raw,   bins=50, alpha=0.55, color='#e74c3c',
            label=f'Raw  (N={len(ip_raw)})', edgecolor='none')
    ax.hist(ip_clean, bins=50, alpha=0.65, color='#2ecc71',
            label=f'Clean (N={len(ip_clean)})', edgecolor='none')
    ax.set_xlabel(f"Chargeability ({meta['ip_unit']})", fontsize=9)
    ax.set_ylabel('Frekuensi', fontsize=9)
    ax.set_title(f"QC - Distribusi IP Sebelum vs Sesudah | {meta['title']}",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('QC_distribusi_IP.png', dpi=150, bbox_inches='tight')
    plt.show()
print("Plot distribusi disimpan.")


# -- Plot QC pseudosection comparison --
fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor='white')
fig.suptitle(f"QC - Pseudosection Sebelum vs Sesudah | {meta['title']}",
             fontsize=12, fontweight='bold')

x_raw       = raw[:, 0]
z_raw       = pseudo_depth(raw[:, 2], raw[:, 1], meta['array_code'])
rho_raw_plt = raw[:, 3]
removed_idx = np.where(~qc_mask)[0]
x_rem = raw[removed_idx, 0]
z_rem = pseudo_depth(raw[removed_idx, 2], raw[removed_idx, 1], meta['array_code'])

xi = np.linspace(x_raw.min(), x_raw.max(), 350)
zi = np.linspace(z_raw.min(), z_raw.max(), 100)
XI, ZI = np.meshgrid(xi, zi)

VI  = griddata((x_raw, z_raw), rho_raw_plt, (XI, ZI), method='linear')
norm_g  = mcolors.Normalize(vmin=np.nanpercentile(rho_raw_plt, 2),
                              vmax=np.nanpercentile(rho_raw_plt, 98))

cf1 = axes[0].contourf(XI, ZI, VI, levels=50, cmap='turbo', norm=norm_g, alpha=0.8)
axes[0].scatter(x_raw, z_raw, c=rho_raw_plt, cmap='turbo', norm=norm_g,
                s=12, marker='v', edgecolors='none', alpha=0.6, zorder=3)
if len(x_rem) > 0:
    axes[0].scatter(x_rem, z_rem, c='red', s=60, marker='x',
                    linewidths=1.5, zorder=5, label=f'Dihapus ({len(x_rem)})')
    axes[0].legend(fontsize=8, loc='upper right')
plt.colorbar(cf1, ax=axes[0], label='rho_a (ohm.m)', shrink=0.85)
axes[0].invert_yaxis()
axes[0].set_ylabel('Pseudo-depth (m)', fontsize=9)
axes[0].set_title(f"SEBELUM QC - {len(rho_raw_plt)} datum", fontsize=10)
axes[0].set_xlim(x_raw.min()-2, x_raw.max()+2)
axes[0].grid(True, alpha=0.25, linestyle='--')
axes[0].contour(XI, ZI, VI, levels=10, colors='k', linewidths=0.3, alpha=0.3)

VI2     = griddata((x_pos, z_pse), rhoa, (XI, ZI), method='linear')
norm_c  = mcolors.Normalize(vmin=np.nanpercentile(rhoa, 2),
                              vmax=np.nanpercentile(rhoa, 98))
cf2 = axes[1].contourf(XI, ZI, VI2, levels=50, cmap='turbo', norm=norm_c, alpha=0.8)
axes[1].scatter(x_pos, z_pse, c=rhoa, cmap='turbo', norm=norm_c,
                s=12, marker='v', edgecolors='none', alpha=0.6, zorder=3)
elec_pos = np.unique(np.round(x_pos, 4))
axes[1].plot(elec_pos, np.zeros(len(elec_pos)), 'v', color='black',
             markersize=4, zorder=5, label='Elektroda')
axes[1].legend(fontsize=8, loc='upper right')
plt.colorbar(cf2, ax=axes[1], label='rho_a (ohm.m)', shrink=0.85)
axes[1].invert_yaxis()
axes[1].set_xlabel('Jarak (m)', fontsize=9)
axes[1].set_ylabel('Pseudo-depth (m)', fontsize=9)
axes[1].set_title(
    f"SESUDAH QC - {len(rhoa)} datum valid ({qc_report['pct_removed']:.1f}% dihapus)",
    fontsize=10)
axes[1].set_xlim(x_raw.min()-2, x_raw.max()+2)
axes[1].grid(True, alpha=0.25, linestyle='--')
axes[1].contour(XI, ZI, VI2, levels=10, colors='k', linewidths=0.3, alpha=0.3)
plt.tight_layout()
plt.savefig('QC_pseudosection_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*58)
print("  RINGKASAN QC - Lintasan 6")
print("="*58)
print(f"  Data awal             : {qc_report['total']:>5d} datum")
for reason, count in qc_report['removed_by'].items():
    print(f"  Dihapus [{reason:<34s}]: {count:>4d}")
print("-"*58)
print(f"  Total dihapus         : {qc_report['n_removed']:>5d} datum ({qc_report['pct_removed']:.1f}%)")
print(f"  Data bersih tersisa   : {qc_report['n_valid']:>5d} datum")
print("="*58)


# ============================================================
# 3. ERT - ANALISIS RESISTIVITAS
# ============================================================

# 3.1 Pseudosection Apparent Resistivity
fig, ax = plt.subplots(figsize=(14, 4.5), facecolor='white')
plot_pseudosection(
    x=x_pos, z=z_pse, values=rhoa, ax=ax,
    title=(f"Pseudosection - Apparent Resistivity (ohm.m) | "
           f"{meta['title']} | {meta['array_name']} | a={meta['unit_spacing']}m"),
    cbar_label='Apparent Resistivity (ohm.m)',
    cmap='Spectral_r', log_scale=False
)
ax.set_xlabel('Jarak (m)', fontsize=9)
ax.text(0.01, 0.05,
        f"rhoa: {rhoa.min():.2f}-{rhoa.max():.2f} ohm.m | "
        f"median: {np.median(rhoa):.2f} ohm.m | N={len(rhoa)}",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
plt.tight_layout()
plt.savefig('pseudosection_ERT.png', dpi=150, bbox_inches='tight')
plt.show()
print("Pseudosection ERT disimpan: pseudosection_ERT.png")


# 3.2 ERT Inversion
print("\nMembangun DataContainerERT...")
scheme_ert = build_ert_container(meta, data_clean, include_ip=False)
print(f"  Sensors   : {scheme_ert.sensorCount()}")
print(f"  Data      : {scheme_ert.size()}")
print(f"  rhoa range: {min(scheme_ert['rhoa']):.3f} - {max(scheme_ert['rhoa']):.3f} ohm.m")

# Adaptive error model
rhoa_arr  = np.array(scheme_ert['rhoa'])
mad_e     = np.median(np.abs(rhoa_arr - np.median(rhoa_arr)))
noise_pct = float(np.clip(1.4826 * mad_e / np.median(rhoa_arr), 0.05, 0.20))
print(f"\nEstimasi noise dari data  : {noise_pct*100:.1f}%")
err_model = noise_pct * rhoa_arr + 0.001
scheme_ert.set('err', pg.Vector(err_model.tolist()))

# L-curve: cari lambda optimal
print("\nMencari lambda optimal (L-curve)...")
lambdas = [1, 2, 5, 10, 20, 50, 100]
chi2s   = []
for lam_test in lambdas:
    mgr_t = ert.ERTManager(scheme_ert)
    mgr_t.invert(lam=lam_test, verbose=False, robustData=True, maxIter=15)
    chi2s.append(mgr_t.inv.chi2())
    print(f"  lam={lam_test:>4d}  ->  chi2={chi2s[-1]:.3f}")

best_lam = lambdas[-1]
for lam_test, c2 in zip(lambdas, chi2s):
    if c2 <= 2.0:
        best_lam = lam_test
        break
    best_lam = lam_test
print(f"\nLambda terpilih : {best_lam}")
print(f"Chi2 estimasi   : {chi2s[lambdas.index(best_lam)]:.3f}")

# ERT Inversion final
print("\nMemulai ERT inversion final...")
mgr_ert = ert.ERTManager(scheme_ert)
inv_ert  = mgr_ert.invert(
    lam=best_lam, lambdaFactor=0.8,
    verbose=True, robustData=True, maxIter=20
)

chi2_ert  = mgr_ert.inv.chi2()
model_ert = np.array(mgr_ert.paraModel(inv_ert))

print(f"\n{'='*52}")
print(f"  ERT Inversion selesai")
print(f"  Lambda dipakai : {best_lam}")
print(f"  Chi2           : {chi2_ert:.4f}")
if chi2_ert <= 1.0:
    print("  Konvergen dengan baik (chi2 <= 1)")
elif chi2_ert <= 2.0:
    print("  Fit cukup baik (1 < chi2 <= 2)")
else:
    print("  chi2 > 2: pertimbangkan efek 3D atau noise data tinggi")
print(f"  Model rho      : {model_ert.min():.3f} - {model_ert.max():.3f} ohm.m")
print(f"{'='*52}")


# 3.3 Visualisasi Hasil ERT Inversion
fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor='white')
fig.suptitle(f"ERT Inversion | {meta['title']} | {meta['array_name']} | a={meta['unit_spacing']}m",
             fontsize=12, fontweight='bold')

plot_pseudosection(
    x=x_pos, z=z_pse, values=rhoa, ax=axes[0],
    title='Pseudosection - Apparent Resistivity (Terukur)',
    cbar_label='rho_a (ohm.m)', cmap='Spectral_r', log_scale=False
)

mgr_ert.showResult(ax=axes[1], cMap='Spectral_r', logScale=False,
                   cMin=model_ert.min(), cMax=model_ert.max())
axes[1].set_title(
    f"Hasil Inversion - True Resistivity (ohm.m) | lam={best_lam} | chi2={chi2_ert:.3f} | robustData=True",
    fontsize=10)
axes[1].set_xlabel('Jarak (m)', fontsize=9)
axes[1].set_ylabel('Kedalaman (m)', fontsize=9)
axes[1].grid(True, alpha=0.25, linestyle='--')
axes[1].text(0.01, 0.06,
             f"Model rho: {model_ert.min():.2f}-{model_ert.max():.2f} ohm.m | "
             f"median: {np.median(model_ert):.2f} ohm.m",
             transform=axes[1].transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
plt.tight_layout()
plt.savefig('ERT_inversion_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Hasil ERT disimpan: ERT_inversion_result.png")


# ============================================================
# 4. IP - ANALISIS CHARGEABILITY
# ============================================================

if ip_val is None:
    print("Tidak ada data IP. Skip Section 4.")
else:
    print(f"\nData IP tersedia: {len(ip_val)} datum")
    print(f"Satuan          : {meta['ip_name']} ({meta['ip_unit']})")
    print(f"Delay time      : {meta['ip_delay']} s")
    print(f"Integration time: {meta['ip_integration']} s")
    print(f"IP range        : {ip_val.min():.1f} - {ip_val.max():.1f} {meta['ip_unit']}")
    print(f"IP median       : {np.median(ip_val):.1f} {meta['ip_unit']}")
    ip_p99    = np.percentile(ip_val, 99)
    n_outlier = np.sum(ip_val > ip_p99)
    print(f"\nData IP > P99 ({ip_p99:.1f} {meta['ip_unit']}): "
          f"{n_outlier} datum ({n_outlier/len(ip_val)*100:.1f}%)")


# 4.1 Pseudosection Apparent Chargeability
if ip_val is not None:
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor='white')
    plot_pseudosection(
        x=x_pos, z=z_pse, values=ip_val, ax=ax,
        title=(f"Pseudosection - Apparent Chargeability ({meta['ip_unit']}) | "
               f"{meta['title']} | gate: {meta['ip_delay']}-"
               f"{meta['ip_delay']+meta['ip_integration']:.2f} s"),
        cbar_label=f"Chargeability ({meta['ip_unit']})",
        cmap='turbo', log_scale=False
    )
    ax.set_xlabel('Jarak (m)', fontsize=9)
    ip_p5, ip_p95 = np.percentile(ip_val, 5), np.percentile(ip_val, 95)
    ax.text(0.01, 0.05,
            f"IP (5-95 pct): {ip_p5:.1f}-{ip_p95:.1f} {meta['ip_unit']} | N={len(ip_val)}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig('pseudosection_IP.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Pseudosection IP disimpan: pseudosection_IP.png")


# 4.2 IP Inversion (Linearized Siegel)
if ip_val is not None:
    print("\nMemulai IP inversion (Linearized Siegel)...")
    try:
        J_ert = np.array(mgr_ert.fop.jacobian())
        print(f"  Jacobian ERT shape: {J_ert.shape}  (n_data x n_model)")
        jacobian_ok = J_ert.shape[0] > 0 and J_ert.shape[1] > 0
    except Exception as e:
        print(f"  Jacobian tidak tersedia: {e}")
        jacobian_ok = False

    if jacobian_ok:
        rho_model = np.array(mgr_ert.paraModel(inv_ert))
        S = J_ert / rho_model[np.newaxis, :]
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        phi_a      = ip_val.copy()
        valid_mask = (phi_a > 0) & (phi_a < np.percentile(phi_a, 99))
        S_valid    = S[valid_mask, :]
        phi_valid  = phi_a[valid_mask]
        print(f"  Data IP valid untuk inversion: {valid_mask.sum()}/{len(phi_a)}")

        lam_ip  = 1e-2
        n_model = S_valid.shape[1]
        StS     = S_valid.T @ S_valid
        Stphi   = S_valid.T @ phi_valid
        m_ip    = np.linalg.solve(StS + lam_ip * np.eye(n_model), Stphi)
        m_ip    = np.clip(m_ip, 0, None)

        print(f"  IP model range  : {m_ip.min():.3f} - {m_ip.max():.3f} ms")
        print(f"  IP model median : {np.median(m_ip):.3f} ms")
        print("  IP inversion berhasil!")
    else:
        m_ip = None
        print("  IP inversion dilewati (Jacobian tidak tersedia).")
else:
    jacobian_ok = False
    m_ip = None


# 4.3 Visualisasi Hasil IP Inversion
if ip_val is not None and jacobian_ok and m_ip is not None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor='white')
    fig.suptitle(f"IP Analysis | {meta['title']} | {meta['array_name']}",
                 fontsize=12, fontweight='bold')

    plot_pseudosection(
        x=x_pos, z=z_pse, values=ip_val, ax=axes[0],
        title='Pseudosection - Apparent Chargeability (Terukur)',
        cbar_label=f"phi_a ({meta['ip_unit']})", cmap='turbo', log_scale=False
    )

    pg.show(mgr_ert.paraDomain, pg.Vector(m_ip), ax=axes[1],
            label='Intrinsic Chargeability (ms)',
            cMap='Spectral_r', logScale=False, hold=True)
    axes[1].set_title(
        f"Hasil IP Inversion - Intrinsic Chargeability (ms) | Siegel | lam={lam_ip}",
        fontsize=10)
    axes[1].set_xlabel('Jarak (m)', fontsize=9)
    axes[1].set_ylabel('Kedalaman (m)', fontsize=9)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    axes[1].text(0.01, 0.06,
                 f"m_IP: {m_ip.min():.2f}-{m_ip.max():.2f} ms | "
                 f"median: {np.median(m_ip):.2f} ms",
                 transform=axes[1].transAxes, fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig('IP_inversion_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Hasil IP disimpan: IP_inversion_result.png")


# ============================================================
# 5. VISUALISASI GABUNGAN ERT + IP
# ============================================================
n_panels = 3 if (ip_val is not None and jacobian_ok and m_ip is not None) else 2
fig, axes = plt.subplots(n_panels, 1,
                          figsize=(14, n_panels*3.8 + 1), facecolor='white')
fig.suptitle(
    f"ERT + IP Combined | {meta['title']} | {meta['array_name']} | a={meta['unit_spacing']}m",
    fontsize=12, fontweight='bold', y=0.99)

plot_pseudosection(
    x=x_pos, z=z_pse, values=rhoa, ax=axes[0],
    title='[ERT] Pseudosection - Apparent Resistivity (Terukur)',
    cbar_label='rho_a (ohm.m)', cmap='Spectral_r', log_scale=False
)
axes[0].set_xlabel('')

mgr_ert.showResult(ax=axes[1], cMap='Spectral_r', logScale=False)
axes[1].set_title(
    f"[ERT] Inversion - True Resistivity (ohm.m) | lam={best_lam} | chi2={chi2_ert:.3f}",
    fontsize=10)
axes[1].set_xlabel('Jarak (m)' if n_panels == 2 else '', fontsize=9)
axes[1].set_ylabel('Kedalaman (m)', fontsize=9)

if n_panels == 3:
    pg.show(mgr_ert.paraDomain, pg.Vector(m_ip), ax=axes[2],
            label='Intrinsic Chargeability (ms)',
            cMap='Spectral_r', logScale=False, hold=True)
    axes[2].set_title(
        f"[IP] Inversion - Intrinsic Chargeability (ms) | Siegel | lam={lam_ip}",
        fontsize=10)
    axes[2].set_xlabel('Jarak (m)', fontsize=9)
    axes[2].set_ylabel('Kedalaman (m)', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('ERT_IP_combined.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot gabungan disimpan: ERT_IP_combined.png")


# ============================================================
# 6. EXPORT MODEL KE CSV
# ============================================================
import pandas as pd

cell_centers = np.array([mgr_ert.paraDomain.cell(i).center()
                          for i in range(mgr_ert.paraDomain.cellCount())])
x_centers = cell_centers[:, 0]
z_centers = cell_centers[:, 1]

df_ert = pd.DataFrame({
    'x_m'               : x_centers,
    'depth_m'           : -z_centers,
    'resistivity_ohm_m' : model_ert
})
df_ert.to_csv('model_ERT_Lintasan6.csv', index=False, float_format='%.4f')
print(f"\nModel ERT disimpan: model_ERT_Lintasan6.csv ({len(df_ert)} sel)")

if ip_val is not None and jacobian_ok and m_ip is not None:
    df_ip = pd.DataFrame({
        'x_m'              : x_centers,
        'depth_m'          : -z_centers,
        'chargeability_ms' : m_ip
    })
    df_ip.to_csv('model_IP_Lintasan6.csv', index=False, float_format='%.4f')
    print(f"Model IP disimpan: model_IP_Lintasan6.csv ({len(df_ip)} sel)")

print("\n" + "="*52)
print("FILE OUTPUT YANG DIHASILKAN:")
print("="*52)
print("  QC_distribusi_rhoa.png")
if meta['ip_flag'] == 1:
    print("  QC_distribusi_IP.png")
print("  QC_pseudosection_comparison.png")
print("  pseudosection_ERT.png")
print("  ERT_inversion_result.png")
if ip_val is not None:
    print("  pseudosection_IP.png")
    if m_ip is not None:
        print("  IP_inversion_result.png")
print("  ERT_IP_combined.png")
print("  model_ERT_Lintasan6.csv")
if ip_val is not None and m_ip is not None:
    print("  model_IP_Lintasan6.csv")
print("="*52)
