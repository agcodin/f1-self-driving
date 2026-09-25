"""Build scale-accurate 3D track files for the sim.

Geometry + widths: TUM racetrack-database centerlines (metres, LGPL-3.0).
Georeferencing: rigid ICP fit (rotation + translation, no scaling) of the TUM
centerline onto OpenStreetMap highway=raceway ways.
Elevation: OpenTopoData DEMs sampled along the georeferenced centerline
(EU-DEM 25 m for Europe, USGS NED 10 m for the US), then smoothed.

Output: ../tracks.js  (window.TRACKS = {...})
"""
import json, math, os, time, urllib.request, urllib.parse
import numpy as np

MAX_ICP_ERROR = 5.0  # metres; above this the georeferencing is not usable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

TRACKS = {
    "monza":  dict(csv="Monza.csv",  osm="osm_monza.json",  dem="eudem25m",
                   name="Autodromo Nazionale Monza", country="Italy"),
    "spa":    dict(csv="Spa.csv",    osm="osm_spa.json",    dem="eudem25m",
                   name="Circuit de Spa-Francorchamps", country="Belgium"),
    "cota":   dict(csv="Austin.csv", osm="osm_austin.json", dem="ned10m",
                   name="Circuit of the Americas", country="USA"),
    # Held out: never trained on, used only to test whether one policy
    # generalises to a circuit it has not seen.
    # More training circuits, for corner-shape diversity: three was not enough
    # to transfer to an unseen layout.
    "zandvoort": dict(csv="Zandvoort.csv", osm="osm_zandvoort.json", dem="eudem25m",
                      name="Circuit Zandvoort", country="Netherlands"),
    "catalunya": dict(csv="Catalunya.csv", osm="osm_catalunya.json", dem="eudem25m",
                      name="Circuit de Barcelona-Catalunya", country="Spain"),
    "hungaroring": dict(csv="Budapest.csv", osm="osm_budapest.json", dem="eudem25m",
                        name="Hungaroring", country="Hungary"),
    "suzuka": dict(csv="Suzuka.csv", osm="osm_suzuka.json", dem="srtm30m",
                   name="Suzuka International Racing Course", country="Japan"),
    "shanghai": dict(csv="Shanghai.csv", osm="osm_shanghai.json", dem="srtm30m",
                     name="Shanghai International Circuit", country="China"),
}


def load_tum(path):
    a = np.loadtxt(path, delimiter=",", comments="#")
    return a[:, :2], a[:, 2], a[:, 3]  # xy, w_right, w_left


def load_osm(path):
    els = json.load(open(path))["elements"]
    lat0 = np.mean([g["lat"] for e in els for g in e.get("geometry", [])])
    lon0 = np.mean([g["lon"] for e in els for g in e.get("geometry", [])])
    R = 6378137.0
    kx = R * math.cos(math.radians(lat0)) * math.pi / 180
    ky = R * math.pi / 180
    pts = []
    for e in els:
        g = e.get("geometry", [])
        xy = np.array([[(p["lon"] - lon0) * kx, (p["lat"] - lat0) * ky] for p in g])
        for i in range(len(xy) - 1):  # densify to ~4 m so NN distance ≈ true distance
            n = max(1, int(np.linalg.norm(xy[i + 1] - xy[i]) / 4))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(xy[i] + t * (xy[i + 1] - xy[i]))
        if len(xy):
            pts.append(xy[-1:])
    return np.vstack(pts), lat0, lon0, kx, ky


def nn(a, b):
    """For each row of a, index + distance of nearest row in b (chunked brute force)."""
    idx = np.empty(len(a), int); dist = np.empty(len(a))
    for s in range(0, len(a), 256):
        d = ((a[s:s + 256, None, :] - b[None, :, :]) ** 2).sum(-1)
        idx[s:s + 256] = d.argmin(1); dist[s:s + 256] = np.sqrt(d.min(1))
    return idx, dist


def icp(src, dst, R, t, iters=40):
    for _ in range(iters):
        cur = src @ R.T + t
        i, d = nn(cur, dst)
        keep = d < np.percentile(d, 90)  # trim outliers (pit lane, chicane variants)
        p, q = src[keep], dst[i[keep]]
        pc, qc = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - pc).T @ (q - qc))
        Rn = (U @ Vt).T
        if np.linalg.det(Rn) < 0:
            Vt[-1] *= -1; Rn = (U @ Vt).T
        R, t = Rn, qc - pc @ Rn.T
    _, d = nn(src @ R.T + t, dst)
    return R, t, np.median(d)


def align(tum, osm):
    """Rigid fit of the TUM centreline onto the OSM raceway geometry.

    The OSM target is filtered against the current fit and the fit redone.
    Venues like Silverstone tag several overlapping layouts (plus the old
    airfield) as highway=raceway, and that spurious geometry pulls a plain ICP
    into a wrong pose -- it settled 32.9 m and 15 degrees out before this.
    """
    sub = tum[:: max(1, len(tum) // 250)]
    best = None
    for deg in range(0, 360, 10):
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        t = osm.mean(0) - sub.mean(0) @ R.T
        R, t, err = icp(sub, osm, R, t, iters=15)
        if best is None or err < best[2]:
            best = (R, t, err)
    R, t, err = icp(tum, osm, best[0], best[1], iters=30)

    target = osm
    for _ in range(3):
        # Keep only OSM points the current fit actually explains, then refit.
        i, d = nn(target, tum @ R.T + t)
        keep = d < max(30.0, np.percentile(d, 20))
        if keep.sum() < 200 or keep.all():
            break
        target = target[keep]
        R, t, err = icp(tum, target, R, t, iters=30)
    return R, t, err


def fetch_elev(lats, lons, dataset):
    out = []
    for s in range(0, len(lats), 100):
        locs = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(lats[s:s + 100], lons[s:s + 100]))
        url = f"https://api.opentopodata.org/v1/{dataset}?" + urllib.parse.urlencode({"locations": locs})
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(url, timeout=60))
                if r["status"] == "OK":
                    break
            except Exception as e:
                print("  retry", e)
            time.sleep(2 + attempt * 2)
        out += [x["elevation"] for x in r["results"]]
        time.sleep(1.1)  # public API: 1 req/s
    return np.array(out, float)


def circ_smooth(v, win_m, ds):
    n = max(1, int(win_m / ds))
    k = np.exp(-0.5 * (np.arange(-3 * n, 3 * n + 1) / n) ** 2); k /= k.sum()
    pad = np.concatenate([v[-3 * n:], v, v[:3 * n]])
    return np.convolve(pad, k, "valid")


def resample_closed(xy, wr, wl, ds):
    loop = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    sn = np.arange(0, L, ds)
    x = np.interp(sn, s, loop[:, 0]); y = np.interp(sn, s, loop[:, 1])
    wr = np.interp(sn, s, np.append(wr, wr[0])); wl = np.interp(sn, s, np.append(wl, wl[0]))
    return np.stack([x, y], 1), wr, wl, L


def main():
    out = {}
    for key, cfg in TRACKS.items():
        xy, wr, wl = load_tum(os.path.join(DATA, cfg["csv"]))
        osm, lat0, lon0, kx, ky = load_osm(os.path.join(DATA, cfg["osm"]))
        R, t, err = align(xy, osm)
        rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        print(f"{key}: ICP median error {err:.1f} m, rotation {rot:.1f} deg")
        # A circuit that did not georeference cannot have trustworthy elevation,
        # because the DEM would be sampled off the track. Drop it rather than
        # ship it. Silverstone fails this: OpenStreetMap tags 20.7 km of
        # raceway there (pit lanes, the Stowe and National layouts) for a 5.9 km
        # circuit, and the fit lands 33 m and 14 degrees out.
        if err > MAX_ICP_ERROR:
            print(f"  SKIPPED: alignment error exceeds {MAX_ICP_ERROR} m")
            continue

        ds = 5.0
        pts, wr, wl, L = resample_closed(xy, wr, wl, ds)
        world = pts @ R.T + t                      # metres, x=east, y=north
        lats = lat0 + world[:, 1] / ky; lons = lon0 + world[:, 0] / kx

        cache = os.path.join(DATA, f"elev_{key}.npy")
        if os.path.exists(cache):
            z = np.load(cache)
        else:
            sample = np.arange(0, len(world), 2)   # DEM every 10 m, interpolate between
            zs = fetch_elev(lats[sample], lons[sample], cfg["dem"])
            z = np.interp(np.arange(len(world)), np.append(sample, len(world)),
                          np.append(zs, zs[0]))
            np.save(cache, z)
        z = circ_smooth(z, 25.0, ds)           # DEM tree/bridge noise → road surface
        grade = np.gradient(z) / ds
        print(f"  length {L:.0f} m, elev {z.min():.1f}..{z.max():.1f} m "
              f"(range {z.max()-z.min():.1f}), max grade {100*np.abs(grade).max():.1f}%")

        world -= world[0]                          # start/finish at origin
        out[key] = dict(
            name=cfg["name"], country=cfg["country"], length=round(float(L), 1), ds=ds,
            lat0=round(float(lats[0]), 6), lon0=round(float(lons[0]), 6),
            x=[round(float(v), 2) for v in world[:, 0]],
            y=[round(float(v), 2) for v in world[:, 1]],
            z=[round(float(v - z.min()), 2) for v in z],
            wr=[round(float(v), 2) for v in wr], wl=[round(float(v), 2) for v in wl],
        )
    with open(os.path.join(HERE, "..", "tracks.js"), "w") as f:
        f.write("// Generated by tools/build_tracks.py — do not edit by hand.\n")
        f.write("(typeof window !== \"undefined\" ? window : globalThis).TRACKS = " + json.dumps(out, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
