# TSP Metaheuristics: Simulated Annealing vs Tabu Search

Tugas kelompok: aplikasi untuk menyelesaikan Travelling Salesman Problem (TSP)
dengan konstruksi greedy, local search (2-opt, 3-opt), Simulated Annealing, dan
Tabu Search, plus perbandingan hasilnya.

**Anggota:** _(isi nama dan NRP)_

## Struktur

```
app.py                      # aplikasi (Streamlit), belum dibuat
tsp_studio/
├── core/                   # City, TSPProblem, metrik jarak
├── algorithms/
│   ├── nearest_neighbor.py, insertion.py     # konstruksi
│   ├── two_opt.py, three_opt.py              # local search
│   ├── simulated_annealing.py, tabu_search.py
│   ├── brute_force.py                        # Held-Karp (eksak, n kecil)
│   ├── teammate.py                           # kontribusi tim (versi dasar), lihat bawah
│   ├── pipeline.py                           # rangkai beberapa solver
│   └── _moves.py                             # helper move 2-opt (SA & Tabu)
├── io/                     # parsers (Excel/CSV/TSPLIB), generators, exporters
├── viz/plots.py            # plot rute & grafik konvergensi
├── evaluation/metrics.py
├── registry.py             # daftar nama algoritma
└── scripts/                # dataset.py, benchmark.py
tests/                      # pytest
datasets/                   # dataset sintetis (JSON)
data/sample_coordinates.xlsx   # contoh file input Excel
docs/                       # paper, ilustrasi tabu, panduan pengguna
presentation/               # PPT
```

## Instalasi

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                  # semua tes harus lulus
python -m tsp_studio.scripts.benchmark  # tabel perbandingan semua algoritma
```

## Cara pakai (API untuk aplikasi)

```python
from tsp_studio import get_algorithm, make_pipeline
from tsp_studio.io.parsers import load_excel
from tsp_studio.viz.plots import plot_route_map, plot_convergence_curve

problem = load_excel("data/sample_coordinates.xlsx")   # -> TSPProblem

# satu algoritma
sa = get_algorithm("simulated_annealing")()
sol = sa.solve(problem, T_max=1000, T_min=1e-3, alpha=0.99,
               max_iter_per_temp=50, seed=42)

# rangkaian: konstruksi -> local search -> metaheuristik
pipe = make_pipeline("nearest_neighbor", "two_opt", "tabu_search")
sol = pipe.solve(problem, tabu_tenure=20, max_iterations=500, seed=42)

sol.tour, sol.length, sol.runtime_sec
sol.metadata["history_best"], sol.metadata["history_current"]
sol.metadata["insights"]      # improvement %, iterasi konvergen, dst.
```

Algoritma terdaftar: `nearest_neighbor`, `nearest_insertion`, `cheapest_insertion`,
`farthest_insertion`, `arbitrary_insertion`, `two_opt`, `three_opt`,
`simulated_annealing`, `tabu_search`, `held_karp_exact` (n <= 18).

Tambahan dari anggota tim (`teammate.py`): `farthest_insertion_far_start`,
`three_opt_swap`, `simulated_annealing_basic`, `tabu_search_basic`.

### Parameter

| Algoritma | Parameter |
|---|---|
| Simulated Annealing | `T_max`, `T_min`, `alpha` (pendinginan geometrik), `max_iter_per_temp`, `seed`, `initial_tour` |
| Tabu Search | `max_iterations`, `tabu_tenure`, `neighborhood_size`, `seed`, `initial_tour`, `deterministic` (seluruh tetangga 2-opt), `record_log` (log per iterasi) |

Catatan: SA dan Tabu butuh matriks jarak **simetris**. Skala `T_max`/`T_min`
sebaiknya disesuaikan dengan ukuran jarak data. Untuk data besar (n >= 100)
mulai dari tour hasil konstruksi (pipeline) dan naikkan jumlah iterasi.

### Format input Excel (`load_excel`)

- **Koordinat:** kolom `x` dan `y` (opsional `id` atau `name`), satu kota per baris.
- **Matriks jarak:** blok n x n angka (boleh dengan baris dan kolom label).

## Kontribusi anggota tim (`tsp_studio/algorithms/teammate.py`)

Versi dasar yang ditulis anggota tim. Logika aslinya dipertahankan, hanya nama
kelas diberi akhiran dan import di `three_opt_swap` diperbaiki.

| Algoritma | Keterangan |
|---|---|
| `farthest_insertion_far_start` | Farthest insertion dengan tour awal = kota awal + kota terjauhnya (versi utama memulai dari kota terdekat). Hasilnya bergantian lebih baik/buruk tergantung dataset. |
| `three_opt_swap` | Local search dengan satu jenis move 3-opt (tukar dua segmen). Lambat untuk n besar (batas benchmark n <= 50). |
| `simulated_annealing_basic` | SA sederhana, menghitung ulang panjang tour penuh tiap tetangga (lebih lambat dari `simulated_annealing`). Parameter: `t_max`, `t_min`, `alpha`, `iter_per_temp`. |
| `tabu_search_basic` | Tabu Search sederhana. Parameter: `max_iter`, `tabu_tenure`, `neighborhood_size`. |

Catatan: SA/Tabu versi dasar memakai random global, jadi `seed` diabaikan dan
hasilnya tidak reproducible; keduanya juga tidak mengisi `insights`/`history_current`
(grafik konvergensi lengkap hanya tersedia dari `simulated_annealing` dan `tabu_search`).
Insertion biasa, `cheapest_insertion`, `nearest_insertion`, dan `Pipeline` versi tim
tidak dimasukkan karena hasilnya identik dengan versi yang sudah ada (diuji pada
11 dataset) atau fungsinya sudah tercakup.

## Status

- [x] Konstruksi (NN, insertion), 2-opt, 3-opt, Held-Karp
- [x] Simulated Annealing, Tabu Search (sebagai `Solver`), loader Excel, plot, tes
- [ ] Aplikasi (`app.py`)
- [ ] Ilustrasi Tabu Search n=6 dengan data tugas 1 (`docs/`): gunakan `deterministic=True, record_log=True`
- [ ] Paper, panduan pengguna, PPT
