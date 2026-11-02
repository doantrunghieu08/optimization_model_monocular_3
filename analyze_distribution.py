import csv

with open('output/brute_force_report.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

col = '% delta_pa-mpjpe'
data = []
for r in rows:
    v = r.get(col, '')
    try:
        val = float(v)
        data.append((r['Cam Master'], r['Cam Slave'], val, float(r.get('pa-mpjpe.all_baseline', 0)), float(r.get('pa-mpjpe.all_method', 0))))
    except (ValueError, TypeError):
        pass

print(f"Total valid pairs: {len(data)}")

high = [x for x in data if x[2] >= 5.0]
med = [x for x in data if 2.0 <= x[2] < 5.0]
low = [x for x in data if 0.0 <= x[2] < 2.0]
neg = [x for x in data if x[2] < 0.0]

print(f"\n--- HIGH GAIN (>= +5%): {len(high)} pairs ---")
for m, s, d, b, f in sorted(high, key=lambda x: -x[2]):
    print(f"  {m} -> {s}: {d:+.2f}% ({b:.2f} -> {f:.2f} mm)")

print(f"\n--- MODERATE GAIN (+2% to +5%): {len(med)} pairs ---")
for m, s, d, b, f in sorted(med, key=lambda x: -x[2]):
    print(f"  {m} -> {s}: {d:+.2f}% ({b:.2f} -> {f:.2f} mm)")

print(f"\n--- LOW GAIN (0% to +2%): {len(low)} pairs ---")
for m, s, d, b, f in sorted(low, key=lambda x: -x[2]):
    print(f"  {m} -> {s}: {d:+.2f}% ({b:.2f} -> {f:.2f} mm)")

print(f"\n--- NEGATIVE (< 0%): {len(neg)} pairs ---")
for m, s, d, b, f in sorted(neg, key=lambda x: x[2]):
    print(f"  {m} -> {s}: {d:+.2f}% ({b:.2f} -> {f:.2f} mm)")

