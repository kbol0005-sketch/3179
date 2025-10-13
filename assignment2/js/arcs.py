from geographiclib.geodesic import Geodesic
import pandas as pd
import json
import math

# --- CONFIG ---
mode = "greatcircle"       # "geodesic" (ellipsoid) or "greatcircle" (sphere)
min_points = 10
max_points = 200
resolution_km = 100        # one point per ~100 km
taper_fraction = 0.25       # fraction of route length for tapering
curvature_factor = 0.1     # 0.1 = 10% bump, 0 = no exaggeration

# --- Custom bins for passenger-based widths ---
bins = bins = [0, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000]
width_values = [15, 30, 40, 50, 70, 90]  # widths (km) for each bin

routes = pd.read_csv("../data/australia_routes_lines.csv")
geod = Geodesic.WGS84
features = []

# --- Assign width bins ---
routes["width_bin"] = pd.cut(routes["Passengers"], bins=bins,
                             labels=False, include_lowest=True)
routes["thickness_km"] = routes["width_bin"].apply(
    lambda b: width_values[int(b)] if pd.notnull(b) else width_values[0]
)

def exaggerate_curve(coords, factor_x=0.0, factor_y=0.1):
    """
    Exaggerate arc curvature smoothly, scaled by route span.

    factor_x = fraction of longitude span for max horizontal bump
               (negative = push west, positive = push east)
    factor_y = fraction of latitude span for max vertical bump
               (positive = lift north, negative = push south)
    """
    if len(coords) < 3 or (factor_x == 0 and factor_y == 0):
        return coords

    lons = [lon for lon, lat in coords]
    lats = [lat for lon, lat in coords]

    span_x = max(lons) - min(lons)
    span_y = max(lats) - min(lats)

    bump_x = span_x * factor_x
    bump_y = span_y * factor_y

    new_coords = []
    n = len(coords)
    for i, (lon, lat) in enumerate(coords):
        t = i / (n - 1)                 # 0 at start, 1 at end
        bump = math.sin(math.pi * t)    # 0 → 1 → 0
        lon_adj = lon + bump * bump_x
        lat_adj = lat + bump * bump_y
        new_coords.append([lon_adj, lat_adj])

    return new_coords




def make_pencil(coords, thickness_km=50, taper_fraction=0.2):
    """Create a tapered 'pencil' polygon along the given coords."""
    thickness_deg = thickness_km / 111.0
    n = len(coords)

    left_side, right_side = [], []

    for i in range(n):
        lon, lat = coords[i]

        # taper factor
        t = i / (n - 1)
        if t < taper_fraction:
            factor = t / taper_fraction
        elif t > (1 - taper_fraction):
            factor = (1 - t) / taper_fraction
        else:
            factor = 1.0

        offset = (thickness_deg / 2) * factor

        # tangent direction
        if i == 0:
            dx = coords[i+1][0] - lon
            dy = coords[i+1][1] - lat
        elif i == n - 1:
            dx = lon - coords[i-1][0]
            dy = lat - coords[i-1][1]
        else:
            dx = coords[i+1][0] - coords[i-1][0]
            dy = coords[i+1][1] - coords[i-1][1]

        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            px, py = 0, 0
        else:
            px = -dy / length
            py = dx / length

        left_side.append([lon + px * offset, lat + py * offset])
        right_side.append([lon - px * offset, lat - py * offset])

    return left_side + right_side[::-1] + [left_side[0]]

# --- MAIN LOOP ---
for _, row in routes.iterrows():
    try:
        origin = row["Origin"]
        dest = row["To"]
        city_pair = row["City-Pair"]

        lat1, lon1 = float(row["origin_lat"]), float(row["origin_lon"])
        lat2, lon2 = float(row["dest_lat"]), float(row["dest_lon"])

        coords = []

        if mode == "geodesic":
            line = geod.InverseLine(lat1, lon1, lat2, lon2)
            distance_km = line.s13 / 1000.0
            npoints = max(min_points, min(max_points, int(distance_km / resolution_km)))

            for i in range(npoints + 1):
                s = i * line.s13 / npoints
                g = line.Position(s, Geodesic.STANDARD)
                coords.append([g["lon2"], g["lat2"]])

        else:  # greatcircle
            lat1r, lon1r = math.radians(lat1), math.radians(lon1)
            lat2r, lon2r = math.radians(lat2), math.radians(lon2)

            d = 2 * math.asin(math.sqrt(
                math.sin((lat2r - lat1r) / 2) ** 2 +
                math.cos(lat1r) * math.cos(lat2r) *
                math.sin((lon2r - lon1r) / 2) ** 2
            ))
            distance_km = d * 6371.0
            npoints = max(min_points, min(max_points, int(distance_km / resolution_km)))

            for i in range(npoints + 1):
                f = i / npoints
                A = math.sin((1 - f) * d) / math.sin(d)
                B = math.sin(f * d) / math.sin(d)

                x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
                y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
                z = A * math.sin(lat1r) + B * math.sin(lat2r)

                lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
                lon = math.degrees(math.atan2(y, x))
                coords.append([lon, lat])

        # --- Apply curvature exaggeration ---
        # --- Special case: PER ↔ BNE nudged north by +3° ---
        if (row["Origin"], row["To"]) in [("MEL","BNE"), ("BNE","MEL")]:
            coords = exaggerate_curve(coords, factor_x=-0.15, factor_y=0.15)
        if (row["Origin"], row["To"]) in [("MEL","OOL"), ("OOL","MEL")]:
            coords = exaggerate_curve(coords, factor_x=-0.10, factor_y=0.1)
        if (row["Origin"], row["To"]) in [("PER","BNE"), ("BNE","PER")]:
            coords = exaggerate_curve(coords, factor_x=-0.05, factor_y=0.8)
        if (row["Origin"], row["To"]) in [("DRW","BNE"), ("BNE","DRW")]:
            coords = exaggerate_curve(coords, factor_x=0.05, factor_y=0.1)
        if (row["Origin"], row["To"]) in [("ADL","MEL"), ("MEL","ADL")]:
            coords = exaggerate_curve(coords, factor_x=-0.05, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("ADL","BNE"), ("BNE","ADL")]:
            coords = exaggerate_curve(coords, factor_x=-0.1, factor_y=0.20)
        if (row["Origin"], row["To"]) in [("ADL","OOL"), ("OOL","ADL")]:
            coords = exaggerate_curve(coords, factor_x=-0.1, factor_y=0.15)
        if (row["Origin"], row["To"]) in [("SYD","BNE"), ("BNE","SYD")]:
            coords = exaggerate_curve(coords, factor_x=0.1, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("DRW","MEL"), ("MEL","DRW")]:
            coords = exaggerate_curve(coords, factor_x=-0.05, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("DRW","SYD"), ("SYD","DRW")]:
            coords = exaggerate_curve(coords, factor_x=-0.05, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("ADL","SYD"), ("SYD","ADL")]:
            coords = exaggerate_curve(coords, factor_x=0, factor_y=0.6)
        if (row["Origin"], row["To"]) in [("ADL","CBR"), ("CBR","ADL")]:
            coords = exaggerate_curve(coords, factor_x=0, factor_y=0.8)
        if (row["Origin"], row["To"]) in [("PER","DRW"), ("DRW","PER")]:
            coords = exaggerate_curve(coords, factor_x=-0.15, factor_y=0.15)
        if (row["Origin"], row["To"]) in [("CNS","BNE"), ("BNE","CNS")]:
            coords = exaggerate_curve(coords, factor_x=0.2, factor_y=0.1)
        if (row["Origin"], row["To"]) in [("CNS","SYD"), ("SYD","CNS")]:
            coords = exaggerate_curve(coords, factor_x=-0.1, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("CNS","MEL"), ("MEL","CNS")]:
            coords = exaggerate_curve(coords, factor_x=-1, factor_y=0)
        if (row["Origin"], row["To"]) in [("MEL","SYD"), ("SYD","MEL")]:
            coords = exaggerate_curve(coords, factor_x=0.20, factor_y=-0.20)
        if (row["Origin"], row["To"]) in [("HBA","SYD"), ("SYD","HBA")]:
            coords = exaggerate_curve(coords, factor_x=0.3, factor_y=-0.15)
        if (row["Origin"], row["To"]) in [("HBA","BNE"), ("BNE","HBA")]:
            coords = exaggerate_curve(coords, factor_x=0.7, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("SYD","OOL"), ("OOL","SYD")]:
            coords = exaggerate_curve(coords, factor_x=0.4, factor_y=-0.1)
        if (row["Origin"], row["To"]) in [("SYD","BNE"), ("BNE","SYD")]:
            coords = exaggerate_curve(coords, factor_x=0., factor_y=-0.)
        if (row["Origin"], row["To"]) in [("CBR","BNE"), ("BNE","CBR")]:
            coords = exaggerate_curve(coords, factor_x=-0.1, factor_y=0.1)
        if (row["Origin"], row["To"]) in [("CBR","OOL"), ("OOL","CBR")]:
            coords = exaggerate_curve(coords, factor_x=-0.05, factor_y=0.05)
        if (row["Origin"], row["To"]) in [("PER","SYD"), ("SYD","PER")]:
            coords = exaggerate_curve(coords, factor_x=0.03, factor_y=1.2)

        # --- Thickness from custom bins ---
        thickness_km = float(row["thickness_km"])

        pencil = make_pencil(coords, thickness_km=thickness_km, taper_fraction=taper_fraction)
        pencil_feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [pencil]},
            "properties": {
                "route": f"{origin}-{dest}",
                "origin": origin,
                "destination": dest,
                "passengers": int(row["Passengers"]),
                "flights": int(row["Aircraft"]),
                "lf": float(row["LF"]),
                "change": float(row["Change"]),
                "distance_km": round(distance_km, 1),
                "npoints": npoints,
                "thickness_km": round(thickness_km, 1),
                "city_pair": city_pair
            }
        }
        features.append(pencil_feature)

    except Exception as e:
        print(f"❌ Error {row['Origin']}-{row['To']}: {e}")

# --- SAVE ---
geojson = {"type": "FeatureCollection", "features": features}
with open("australia_routes_lines.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

print(f"✅ Generated {len(features)} passenger-scaled pencil arcs (curved) and saved to australia_routes_lines.geojson")
