"""Antarmuka Streamlit untuk proyek TSP.
Jalankan dari folder proyek: streamlit run app.py
Integrasi tsp_studio dibatasi pada API yang dicontohkan di README.
"""
from __future__ import annotations

import inspect
import io
import statistics
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from tsp_studio import get_algorithm, make_pipeline
from tsp_studio.io.parsers import load_excel
from tsp_studio.viz.plots import plot_route_map, plot_convergence_curve

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "data" / "sample_coordinates.xlsx"

st.set_page_config(page_title="TSP Optimizer", layout="wide", initial_sidebar_state="expanded")

BEST = "#56B4E9"
CURRENT = "#E69F00"
START = "#009E73"
WARNING = "#D55E00"
TEXT = "#E6EDF3"
GRID = "#49515F"

ALGORITHMS = {
    "Nearest Neighbor": "nearest_neighbor",
    "Nearest Insertion": "nearest_insertion",
    "Cheapest Insertion": "cheapest_insertion",
    "Farthest Insertion": "farthest_insertion",
    "Arbitrary Insertion": "arbitrary_insertion",
    "2-opt": "two_opt",
    "3-opt": "three_opt",
    "Simulated Annealing": "simulated_annealing",
    "Tabu Search": "tabu_search",
    "Held-Karp (eksak)": "held_karp_exact",
}
EXACT = "Held-Karp (eksak)"
CONSTRUCTION = [
    "Nearest Neighbor", "Nearest Insertion", "Cheapest Insertion",
    "Farthest Insertion", "Arbitrary Insertion",
]
LOCAL_SEARCH = ["2-opt", "3-opt"]
META = ["Simulated Annealing", "Tabu Search"]

HELD_KARP_MAX_N = 18
STOCHASTIC_STAGES = {"simulated_annealing", "tabu_search"}
STAGE_KEYS = {
    "simulated_annealing": {"seed", "T_max", "T_min", "alpha", "max_iter_per_temp"},
    "tabu_search": {"seed", "max_iterations", "tabu_tenure", "neighborhood_size"},
}


def init_state() -> None:
    defaults = {"problem": None, "problem_label": "", "results": [], "last_run_key": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def style_figure(fig):
    fig.patch.set_facecolor("#0E1117")
    for ax in fig.axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors=TEXT, labelsize=10)
        for annotation in ax.texts:
            annotation.set_color("#FFFFFF")
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, color=GRID, linestyle=":", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#1B2130")
            legend.get_frame().set_edgecolor(GRID)
            for label in legend.get_texts():
                label.set_color(TEXT)
    fig.tight_layout()
    return fig


def close_figure(fig) -> None:
    plt.close(fig)


def load_excel_bytes(data: bytes, filename: str):
    suffix = Path(filename).suffix or ".xlsx"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(data)
            temp_path = Path(temp.name)
        return load_excel(temp_path)
    except ValueError as exc:
        raise ValueError(f"Format Excel belum sesuai: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            "File tidak dapat dibaca. Pastikan file berformat .xlsx dan berisi kolom x dan y "
            "atau matriks jarak yang valid."
        ) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def generate_excel(pattern: str, n: int, seed: int) -> bytes:
    import random
    rng = random.Random(seed)
    points = []
    if pattern == "Tersebar merata":
        points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    elif pattern == "Berkelompok":
        centers = [(rng.uniform(10, 90), rng.uniform(10, 90)) for _ in range(min(5, n))]
        for i in range(n):
            cx, cy = centers[i % len(centers)]
            points.append((max(0, min(100, rng.gauss(cx, 8))),
                           max(0, min(100, rng.gauss(cy, 8)))))
    else:
        import math
        cols = math.ceil(math.sqrt(n))
        for i in range(n):
            points.append((float(i % cols), float(i // cols)))
    df = pd.DataFrame({
        "name": [f"Kota {i+1}" for i in range(n)],
        "x": [round(p[0], 4) for p in points],
        "y": [round(p[1], 4) for p in points],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


def city_table(problem) -> pd.DataFrame:
    rows = []
    for i, city in enumerate(problem.cities):
        rows.append({
            "ID": getattr(city, "id", i),
            "Kota": getattr(city, "name", None) or f"Kota {i+1}",
            "x": getattr(city, "x", None),
            "y": getattr(city, "y", None),
        })
    return pd.DataFrame(rows)


def is_stochastic(stages: list[str]) -> bool:
    return any(stage in STOCHASTIC_STAGES for stage in stages)


def build_pipelines(selected: list[str]) -> tuple[list[tuple[str, list[str]]], str | None]:
    constructions = [x for x in CONSTRUCTION if x in selected]
    improvements = [x for x in LOCAL_SEARCH if x in selected]
    metas = [x for x in META if x in selected]
    exacts = [x for x in selected if x == EXACT]

    notice = None
    starts = constructions
    if not starts and (improvements or metas):
        starts = ["Nearest Neighbor"]
        notice = (
            "Tidak ada metode konstruksi yang dipilih, sehingga Nearest Neighbor dipakai sebagai "
            "tahap awal dan ikut dijalankan sendiri sebagai pembanding."
        )

    specs = []
    for start in starts:
        specs.append((start, [ALGORITHMS[start]]))
        for step in improvements + metas:
            specs.append((f"{start} → {step}", [ALGORITHMS[start], ALGORITHMS[step]]))
        combined = [start] + improvements + metas
        if len(combined) > 2:
            specs.append((" → ".join(combined), [ALGORITHMS[s] for s in combined]))
    for exact in exacts:
        specs.append((exact, [ALGORITHMS[exact]]))

    unique, seen = [], set()
    for label, stages in specs:
        key = tuple(stages)
        if key not in seen:
            seen.add(key)
            unique.append((label, stages))
    return unique, notice


def runnable_specs(specs, problem):
    if problem is None:
        return specs
    return [
        (label, stages) for label, stages in specs
        if not ("held_karp_exact" in stages and problem.n > HELD_KARP_MAX_N)
    ]


def count_executions(specs, n_runs: int) -> int:
    return sum(n_runs if is_stochastic(stages) else 1 for _, stages in specs)


def recommend_algorithms(problem) -> list[dict[str, str]]:
    n = int(problem.n)
    recommendations: list[dict[str, str]] = []

    if n <= HELD_KARP_MAX_N:
        recommendations.append({
            "priority": "Acuan kualitas terbaik",
            "pipeline": EXACT,
            "reason": (
                f"Ada {n} kota. Held-Karp dapat mencari solusi optimal secara eksak untuk ukuran ini, "
                "sehingga cocok sebagai acuan untuk menilai kualitas hasil metode heuristik."
            ),
        })
    if n <= 50:
        recommendations.append({
            "priority": "Pilihan awal yang seimbang",
            "pipeline": "Nearest Neighbor → 2-opt",
            "reason": (
                "Nearest Neighbor membangun rute awal dengan cepat, lalu 2-opt memperbaiki urutan "
                "kunjungan. Ini pilihan baseline yang sederhana dan umumnya efisien untuk diuji."
            ),
        })
    if n <= 120:
        recommendations.append({
            "priority": "Eksplorasi kualitas lebih lanjut",
            "pipeline": "Nearest Neighbor → 2-opt → Tabu Search",
            "reason": (
                "Tabu Search membantu mengeksplorasi solusi tetangga dan mengurangi risiko terjebak "
                "terlalu cepat pada solusi lokal. Waktu proses bertambah dan hasil tetap perlu dibandingkan."
            ),
        })
        recommendations.append({
            "priority": "Alternatif metaheuristik",
            "pipeline": "Nearest Neighbor → Simulated Annealing",
            "reason": (
                "Simulated Annealing sesekali menerima langkah yang lebih buruk untuk memperluas eksplorasi. "
                "Cocok dijadikan pembanding terhadap Tabu Search dengan seed dan anggaran komputasi yang sama."
            ),
        })
    else:
        recommendations.append({
            "priority": "Untuk data besar",
            "pipeline": "Nearest Neighbor → 2-opt",
            "reason": (
                f"Ada {n} kota. Mulai dari metode yang relatif ringan; uji Tabu Search setelahnya bila "
                "waktu komputasi mencukupi. 3-opt dan Held-Karp berpotensi mahal pada ukuran ini."
            ),
        })
        recommendations.append({
            "priority": "Peningkatan bertahap",
            "pipeline": "Nearest Neighbor → Tabu Search",
            "reason": (
                "Gunakan sebagai eksperimen lanjutan dengan batas iterasi yang terkontrol, lalu bandingkan "
                "panjang rute dan waktu eksekusi terhadap baseline."
            ),
        })

    has_coordinates = bool(getattr(problem, "cities", None)) and all(
        getattr(city, "x", None) is not None and getattr(city, "y", None) is not None
        for city in problem.cities
    )
    if not has_coordinates:
        recommendations = [
            {**item, "reason": item["reason"] + " Panduan ini tetap dapat dipakai untuk matriks jarak."}
            for item in recommendations
        ]
    return recommendations


def call_solve(solver, problem, params: dict):
    signature = inspect.signature(solver.solve)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return solver.solve(problem, **params)
    accepted = {k: v for k, v in params.items() if k in signature.parameters}
    return solver.solve(problem, **accepted)


def run_one(label: str, stages: list[str], problem, params: dict):
    if len(stages) == 1:
        solver = get_algorithm(stages[0])()
        keys = STAGE_KEYS.get(stages[0], {"seed"})
        return call_solve(solver, problem, {k: v for k, v in params.items() if k in keys})
    solver = make_pipeline(*stages, custom_name=label)
    return solver.solve(problem, **params)


def run_pipeline(label, stages, problem, params, n_runs, base_seed, on_step):
    total = n_runs if is_stochastic(stages) else 1
    solutions, errors = [], []
    for i in range(total):
        try:
            solutions.append(run_one(label, stages, problem, {**params, "seed": base_seed + i}))
        except Exception as exc:
            errors.append(str(exc))
        on_step()
    if not solutions:
        return {"label": label, "stages": stages, "solution": None, "solutions": [],
                "runs": 0, "error": errors[0] if errors else "Tidak ada hasil."}
    best = min(solutions, key=lambda s: s.length)
    error = f"{len(errors)} dari {total} run gagal: {errors[0]}" if errors else None
    return {"label": label, "stages": stages, "solution": best, "solutions": solutions,
            "runs": len(solutions), "error": error}


def summarize(result) -> dict:
    solutions = result["solutions"]
    lengths = [float(s.length) for s in solutions]
    times = [float(s.runtime_sec) for s in solutions]
    improvements = []
    for sol in solutions:
        value = ((sol.metadata or {}).get("insights") or {}).get("improvement_pct")
        try:
            if value is not None:
                improvements.append(float(value))
        except (TypeError, ValueError):
            pass
    return {
        "mean": statistics.fmean(lengths),
        "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "best": min(lengths),
        "time": statistics.fmean(times),
        "improvement": statistics.fmean(improvements) if improvements else float("nan"),
    }


init_state()

with st.sidebar:
    st.header("TSP Optimizer")
    st.caption("Alur kerja")
    with st.expander("1. Data", expanded=True):
        source = st.radio(
            "Sumber data",
            ["Data contoh", "Unggah Excel", "Data acak"],
            help="Pilih data contoh, unggah berkas sendiri, atau buat koordinat sintetis.",
        )
        if source == "Data contoh":
            st.caption("Menggunakan data/sample_coordinates.xlsx.")
            if st.button("Muat data contoh", help="Baca dataset contoh dari folder proyek.", width="stretch"):
                try:
                    if not SAMPLE_FILE.exists():
                        raise ValueError("File data/sample_coordinates.xlsx tidak ditemukan di folder proyek.")
                    st.session_state["problem"] = load_excel(SAMPLE_FILE)
                    st.session_state["problem_label"] = "Data contoh (sample_coordinates.xlsx)"
                    st.session_state["results"] = []
                    st.session_state["last_run_key"] = None
                except ValueError as exc:
                    st.error(str(exc))
        elif source == "Unggah Excel":
            uploaded = st.file_uploader(
                "Berkas Excel (.xlsx)",
                type=["xlsx"],
                help="Gunakan kolom x dan y untuk koordinat, atau matriks jarak yang didukung parser.",
            )
            if uploaded:
                try:
                    preview_problem = load_excel_bytes(uploaded.getvalue(), uploaded.name)
                    st.caption(f"Pratinjau: {preview_problem.n} kota terbaca.")
                    st.dataframe(city_table(preview_problem), hide_index=True, width="stretch")
                    if st.button("Gunakan berkas ini", width="stretch"):
                        st.session_state["problem"] = preview_problem
                        st.session_state["problem_label"] = uploaded.name
                        st.session_state["results"] = []
                        st.session_state["last_run_key"] = None
                except ValueError as exc:
                    st.error(str(exc))
        else:
            pattern = st.selectbox(
                "Pola koordinat",
                ["Tersebar merata", "Berkelompok", "Pola kisi"],
                help="Pola menentukan bentuk sebaran lokasi yang dibuat.",
            )
            n_random = st.slider("Jumlah kota", 3, 200, 25, help="Jumlah lokasi yang akan dibuat.")
            random_seed = st.number_input("Seed data", min_value=0, max_value=999999, value=42,
                                          help="Seed sama menghasilkan data acak yang sama.")
            if st.button("Buat data", width="stretch"):
                try:
                    data = generate_excel(pattern, int(n_random), int(random_seed))
                    st.session_state["problem"] = load_excel_bytes(data, "data_acak.xlsx")
                    st.session_state["problem_label"] = f"{pattern}, {n_random} kota (seed {random_seed})"
                    st.session_state["results"] = []
                    st.session_state["last_run_key"] = None
                except ValueError as exc:
                    st.error(str(exc))

    with st.expander("2. Algoritma", expanded=True):
        selected = st.multiselect(
            "Pilih metode untuk dibandingkan",
            options=list(ALGORITHMS),
            default=["Nearest Neighbor", "2-opt", "Tabu Search"],
            help="Pilih beberapa metode. Setiap metode konstruksi dijalankan sendiri sebagai pembanding, "
                 "lalu digabung dengan pencarian lokal atau metaheuristik.",
        )
        st.caption("Pipeline dibuat dari metode konstruksi, lalu metode perbaikan yang dipilih.")

    with st.expander("3. Parameter", expanded=True):
        seed = st.number_input("Seed algoritma", min_value=0, max_value=999999, value=42,
                               help="Seed awal. Run ke-i memakai seed awal + i.")
        n_runs = st.number_input("Jumlah run (metode stokastik)", min_value=1, max_value=50, value=5,
                                 help="Simulated Annealing dan Tabu Search dijalankan sebanyak ini dengan seed "
                                      "berurutan. Metode deterministik dijalankan sekali.")
        sa_selected = "Simulated Annealing" in selected
        tabu_selected = "Tabu Search" in selected
        if sa_selected:
            st.markdown("**Simulated Annealing**")
            t_max = st.number_input("T_max", min_value=0.000001, value=100.0,
                                    help="Suhu awal; nilai lebih besar memberi eksplorasi awal lebih luas.")
            t_min = st.number_input("T_min", min_value=0.0000001, value=0.1, format="%.7f",
                                    help="Suhu akhir; pencarian berhenti saat suhu turun ke nilai ini.")
            alpha = st.number_input("alpha", min_value=0.5, max_value=0.99999, value=0.99, step=0.01,
                                    help="Faktor pendinginan setiap tahap; harus kurang dari 1.")
            sa_iter = st.number_input("max_iter_per_temp", min_value=1, max_value=10000, value=50,
                                      help="Jumlah percobaan pada setiap tingkat suhu.")
        if tabu_selected:
            st.markdown("**Tabu Search**")
            tabu_iterations = st.number_input("max_iterations", min_value=1, max_value=100000,
                                              value=500, help="Batas jumlah iterasi pencarian.")
            tabu_tenure = st.number_input("tabu_tenure", min_value=1, max_value=1000, value=20,
                                          help="Berapa lama langkah yang baru dipakai diingat sebagai tabu.")
            tabu_neighborhood = st.number_input("neighborhood_size", min_value=1, max_value=10000,
                                                value=60, help="Jumlah kandidat tetangga yang diperiksa per iterasi.")
        st.caption("Parameter hanya ditampilkan untuk metode yang dipilih.")

    with st.expander("4. Jalankan", expanded=True):
        st.caption("Periksa data dan metode, lalu jalankan perbandingan.")
        run_button = st.button("Jalankan", type="primary", width="stretch",
                               help="Jalankan semua pipeline yang dipilih dan simpan hasilnya.")

st.title("Optimasi Rute TSP")

problem = st.session_state["problem"]
results = st.session_state["results"]

if problem is not None:
    st.caption(f"Data aktif: {st.session_state['problem_label']} · {problem.n} kota")
else:
    st.caption("Pilih data di panel kiri untuk memulai.")

if problem is not None and problem.n > 50 and "3-opt" in selected:
    st.warning("3-opt dapat berjalan lambat untuk lebih dari 50 kota. Pertimbangkan data yang lebih kecil.")
if problem is not None and problem.n > HELD_KARP_MAX_N and EXACT in selected:
    st.warning(f"Held-Karp (eksak) dibatasi untuk maksimal {HELD_KARP_MAX_N} kota; metode ini tidak akan dijalankan.")

pipeline_specs, start_notice = build_pipelines(selected)
runnable = runnable_specs(pipeline_specs, problem)
if selected and start_notice:
    st.info(start_notice)
if problem is not None and runnable:
    st.caption(
        f"Akan menjalankan {len(runnable)} pipeline, total {count_executions(runnable, int(n_runs))} eksekusi."
    )

if run_button:
    if problem is None:
        st.error("Pilih atau muat data terlebih dahulu di bagian 1. Data.")
    elif not selected:
        st.error("Pilih minimal satu algoritma di bagian 2. Algoritma.")
    else:
        valid_specs = runnable
        new_results = []
        params = {"seed": int(seed)}
        if sa_selected:
            params.update(T_max=float(t_max), T_min=float(t_min), alpha=float(alpha),
                          max_iter_per_temp=int(sa_iter))
        if tabu_selected:
            params.update(max_iterations=int(tabu_iterations), tabu_tenure=int(tabu_tenure),
                          neighborhood_size=int(tabu_neighborhood))
        total_steps = max(count_executions(valid_specs, int(n_runs)), 1)
        step = {"n": 0}
        progress = st.progress(0, text="Menyiapkan perhitungan")
        for label, stages in valid_specs:
            def tick(label=label):
                step["n"] += 1
                progress.progress(min(step["n"] / total_steps, 1.0),
                                  text=f"{label} ({step['n']}/{total_steps})")
            new_results.append(
                run_pipeline(label, stages, problem, params, int(n_runs), int(seed), tick)
            )
        progress.progress(1.0, text="Selesai")
        st.session_state["results"] = new_results
        st.session_state["last_run_key"] = (st.session_state["problem_label"], problem.n, int(seed))
        results = new_results
        if not valid_specs:
            st.warning("Tidak ada metode yang dapat dijalankan dengan batas ukuran data saat ini.")

tabs = st.tabs(["Panduan", "Rute", "Konvergensi", "Perbandingan", "Detail"])
valid_results = [r for r in results if r.get("solution") is not None]
errors = [r for r in results if r.get("error")]

with tabs[0]:
    st.subheader("Panduan heuristik berdasarkan ukuran data")
    st.write(
        "Bagian ini hanya aturan praktis berdasarkan jumlah kota, bukan hasil eksperimen pada data Anda. "
        "Pilih metode sendiri, lalu buktikan mana yang lebih baik di tab Perbandingan."
    )
    if problem is None:
        st.info("Muat data terlebih dahulu di panel kiri agar panduan dapat disesuaikan dengan ukuran masalah.")
    else:
        st.caption(f"Analisis untuk {problem.n} kota.")
        for item in recommend_algorithms(problem):
            with st.container(border=True):
                st.caption(item["priority"])
                st.markdown(f"**{item['pipeline']}**")
                st.write(item["reason"])
                if item["pipeline"] == EXACT:
                    st.caption(f"Tersedia di aplikasi jika jumlah kota tidak lebih dari {HELD_KARP_MAX_N}.")
        st.info(
            "Saran praktis: jalankan dua atau tiga kandidat dengan beberapa run dan parameter yang sama, "
            "lalu bandingkan rata-rata, simpangan baku, dan waktu eksekusi pada tab Perbandingan."
        )

with tabs[1]:
    if problem is None or not valid_results:
        st.info("Pilih data di panel kiri, pilih algoritma, lalu tekan Jalankan.")
    else:
        choices = [r["label"] for r in valid_results]
        route_choice = st.selectbox("Pilih pipeline untuk melihat rute", choices,
                                    help="Pilih hasil yang ingin ditampilkan pada peta.")
        chosen_route = next(r for r in valid_results if r["label"] == route_choice)
        try:
            fig = plot_route_map(problem, chosen_route["solution"].tour, title=f"Rute — {route_choice}")
            style_figure(fig)
            st.pyplot(fig, width="stretch")
            close_figure(fig)
            st.caption(
                f"Rute run terbaik dari {chosen_route['runs']} run: melewati {problem.n} kota, "
                f"panjang {chosen_route['solution'].length:.2f}."
            )
        except ValueError:
            st.warning("Peta rute hanya tersedia untuk data koordinat; dataset ini berupa matriks jarak.")

with tabs[2]:
    convergence_results = [
        r for r in valid_results
        if (r["solution"].metadata or {}).get("history_best")
        and (r["solution"].metadata or {}).get("history_current")
    ]
    if not convergence_results:
        st.info("Grafik konvergensi akan tersedia setelah metode yang mencatat riwayat pencarian dijalankan.")
    else:
        convergence_choice = st.selectbox("Pilih pipeline untuk melihat konvergensi",
                                          [r["label"] for r in convergence_results],
                                          help="Pilih hasil metaheuristik yang memiliki riwayat pencarian.")
        conv = next(r for r in convergence_results if r["label"] == convergence_choice)
        meta = conv["solution"].metadata or {}
        fig = plot_convergence_curve(meta["history_best"], meta["history_current"],
                                     title=f"Konvergensi — {convergence_choice}")
        ax = fig.axes[0]
        for line in ax.lines:
            if "current" in line.get_label().lower():
                line.set_color(CURRENT)
                line.set_linestyle("--")
                line.set_marker("x")
                line.set_markevery(max(1, len(line.get_xdata()) // 12))
                line.set_label("Solusi saat ini yang diterima (garis putus-putus)")
            else:
                line.set_color(BEST)
                line.set_linestyle("-")
                line.set_marker("o")
                line.set_markevery(max(1, len(line.get_xdata()) // 12))
                line.set_label("Rute terbaik sejauh ini (garis penuh)")
        ax.legend()
        style_figure(fig)
        st.pyplot(fig, width="stretch")
        close_figure(fig)
        st.caption(
            "Garis penuh adalah rute terbaik sejauh ini dan tidak pernah naik. Garis putus-putus adalah solusi "
            "saat ini yang diterima algoritma, bukan kandidat yang sedang diuji. Garis ini bisa lebih buruk "
            "dari yang terbaik: Simulated Annealing menerima langkah yang lebih buruk dengan peluang tertentu, "
            f"dan Tabu Search harus berpindah ke tetangga terbaik yang tidak tabu meski lebih buruk. "
            f"Grafik berasal dari run terbaik dari {conv['runs']} run."
        )

with tabs[3]:
    if not valid_results:
        st.info("Jalankan setidaknya satu pipeline untuk melihat tabel perbandingan.")
    else:
        summaries = [(r, summarize(r)) for r in valid_results]
        overall_best = min(s["best"] for _, s in summaries)
        exact_length = next((s["best"] for r, s in summaries if "held_karp_exact" in r["stages"]), None)
        ordered = sorted(summaries, key=lambda item: item[1]["mean"])
        rows = []
        for rank, (result, summary) in enumerate(ordered, 1):
            rows.append({
                "Peringkat": rank,
                "Pipeline": result["label"],
                "Jumlah run": result["runs"],
                "Panjang rata-rata": summary["mean"],
                "Simpangan baku": summary["std"],
                "Panjang terbaik": summary["best"],
                "Waktu rata-rata (detik)": summary["time"],
                "Perbaikan rata-rata (%)": summary["improvement"],
                "Selisih dari terbaik yang ditemukan (%)": (
                    (summary["mean"] - overall_best) / overall_best * 100 if overall_best else 0.0
                ),
                "Gap optimal (%)": (
                    (summary["mean"] - exact_length) / exact_length * 100 if exact_length else float("nan")
                ),
            })
        comparison = pd.DataFrame(rows)
        st.dataframe(comparison, hide_index=True, width="stretch",
                     column_config={
                         "Jumlah run": st.column_config.NumberColumn(format="%d"),
                         "Panjang rata-rata": st.column_config.NumberColumn(format="%.2f"),
                         "Simpangan baku": st.column_config.NumberColumn(format="%.2f"),
                         "Panjang terbaik": st.column_config.NumberColumn(format="%.2f"),
                         "Waktu rata-rata (detik)": st.column_config.NumberColumn(format="%.4f"),
                         "Perbaikan rata-rata (%)": st.column_config.NumberColumn(format="%.2f"),
                         "Selisih dari terbaik yang ditemukan (%)": st.column_config.NumberColumn(format="%.2f"),
                         "Gap optimal (%)": st.column_config.NumberColumn(format="%.2f"),
                     })
        st.download_button("Unduh perbandingan CSV", comparison.to_csv(index=False).encode("utf-8"),
                           file_name="perbandingan_tsp.csv", mime="text/csv")
        st.caption(
            f"Tabel membandingkan {len(comparison)} pipeline, diurutkan menurut panjang rute rata-rata. "
            "Metode stokastik (Simulated Annealing, Tabu Search) dijalankan beberapa kali dengan seed berurutan; "
            "metode deterministik dijalankan sekali, sehingga simpangan bakunya 0."
        )
        if exact_length is None:
            st.caption(
                "Selisih dihitung terhadap rute terpendek yang ditemukan pada sesi ini, bukan terhadap optimum "
                f"sejati. Kolom Gap optimal hanya terisi jika Held-Karp ikut dijalankan (maksimal {HELD_KARP_MAX_N} kota)."
            )
        else:
            st.caption(
                f"Gap optimal memakai hasil Held-Karp ({exact_length:.2f}) sebagai optimum. Selisih dari terbaik "
                "yang ditemukan tetap dihitung terhadap rute terpendek di antara semua pipeline."
            )

with tabs[4]:
    if not valid_results:
        st.info("Detail akan muncul setelah perhitungan selesai.")
    else:
        detail_choice = st.selectbox("Pilih pipeline untuk melihat detail",
                                     [r["label"] for r in valid_results],
                                     help="Pilih pipeline untuk melihat urutan kota dan ringkasan algoritma.")
        detail = next(r for r in valid_results if r["label"] == detail_choice)
        sol = detail["solution"]
        summary = summarize(detail)
        col_best, col_mean, col_time = st.columns(3)
        col_best.metric("Panjang terbaik", f"{summary['best']:.2f}")
        col_mean.metric("Rata-rata ± simpangan baku", f"{summary['mean']:.2f} ± {summary['std']:.2f}")
        col_time.metric("Waktu rata-rata", f"{summary['time']:.4f} detik")
        st.caption(
            f"Ringkasan proses dan urutan kunjungan di bawah berasal dari run terbaik dari {detail['runs']} run."
        )
        st.write("**Tahapan pipeline**")
        st.code(" → ".join(detail["stages"]), language=None)
        st.write("**Ringkasan proses**")
        insights = (sol.metadata or {}).get("insights", {})
        if insights:
            insight_df = pd.DataFrame([{"Keterangan": k.replace("_", " ").capitalize(), "Nilai": v}
                                       for k, v in insights.items()])
            st.dataframe(insight_df, hide_index=True, width="stretch")
        else:
            st.caption("Metode ini tidak menyediakan ringkasan insight tambahan.")
        if problem is not None:
            st.write("**Urutan kunjungan**")
            tour = list(sol.tour)
            names = []
            for idx in tour:
                city = problem.cities[idx]
                names.append(str(getattr(city, "name", None) or f"Kota {idx+1}"))
            st.code(" → ".join(names + names[:1]), language=None)
        st.caption(f"Detail untuk pipeline {detail_choice}.")

if errors:
    with st.expander("Lihat pesan metode yang gagal"):
        for result in errors:
            st.warning(f"{result['label']}: {result['error']}")