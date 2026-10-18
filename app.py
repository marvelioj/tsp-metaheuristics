"""Antarmuka Streamlit untuk proyek TSP.
Jalankan dari folder proyek: streamlit run app.py
Integrasi tsp_studio dibatasi pada API yang dicontohkan di README.
"""
from __future__ import annotations

import io
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

# Warna grafik mengikuti palet yang diminta; teks tetap kontras pada tema gelap.
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
CONSTRUCTION = [
    "Nearest Neighbor", "Nearest Insertion", "Cheapest Insertion",
    "Farthest Insertion", "Arbitrary Insertion",
]
LOCAL_SEARCH = ["2-opt", "3-opt"]
META = ["Simulated Annealing", "Tabu Search"]


def init_state() -> None:
    defaults = {"problem": None, "problem_label": "", "results": [], "last_run_key": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def style_figure(fig):
    """Samakan warna latar, label, grid, dan legenda untuk tema gelap."""
    fig.patch.set_facecolor("#0E1117")
    for ax in fig.axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors=TEXT, labelsize=10)
        # Pastikan label/anotasi nama kota pada grafik terbaca di tema gelap.
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
    """Gunakan parser resmi proyek dan berikan pesan yang bisa dipahami."""
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
    """Buat koordinat sintetis sendiri lalu serahkan parsing ke load_excel."""
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


def make_pipeline_names(selected: list[str]) -> list[tuple[str, list[str]]]:
    """Bangun pipeline yang valid: satu konstruksi, lalu optional local search/metaheuristic."""
    labels = []
    constructions = [x for x in selected if x in CONSTRUCTION]
    improvements = [x for x in selected if x in LOCAL_SEARCH]
    metas = [x for x in selected if x in META]
    exacts = [x for x in selected if x == "Held-Karp (eksak)"]
    # Tambahkan pipeline gabungan sesuai urutan kategori: konstruksi -> local search -> metaheuristik.
    # Ini memastikan pilihan Nearest Neighbor + 2-opt + Tabu Search benar-benar diuji sebagai satu pipeline.
    starts = constructions or (["Nearest Neighbor"] if (improvements or metas) else [])
    for start in starts:
        combined = [start] + improvements + metas
        labels.append((" → ".join(combined), [ALGORITHMS[s] for s in combined]))
        # Sertakan juga varian perbaikan individual agar hasilnya dapat dibandingkan.
        for improve in improvements:
            labels.append((f"{start} → {improve}", [ALGORITHMS[start], ALGORITHMS[improve]]))
        for meta in metas:
            labels.append((f"{start} → {meta}", [ALGORITHMS[start], ALGORITHMS[meta]]))
    if not starts:
        for name in improvements + metas:
            labels.append((name, [ALGORITHMS[name]]))
    for exact in exacts:
        labels.append((exact, [ALGORITHMS[exact]]))
    # Hindari pipeline duplikat.
    unique, seen = [], set()
    for label, stages in labels:
        key = tuple(stages)
        if key not in seen:
            seen.add(key)
            unique.append((label, stages))
    return unique


def recommend_algorithms(problem) -> list[dict[str, str]]:
    """Rekomendasi berbasis ukuran masalah dan batas praktis algoritma.

    Ini adalah panduan heuristik, bukan jaminan bahwa algoritma tertentu selalu terbaik.
    """
    n = int(problem.n)
    recommendations: list[dict[str, str]] = []

    if n <= 18:
        recommendations.append({
            "priority": "Acuan kualitas terbaik",
            "pipeline": "Held-Karp (eksak)",
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
            {**item, "reason": item["reason"] + " Rekomendasi ini tetap dapat dipakai untuk matriks jarak."}
            for item in recommendations
        ]
    return recommendations


def run_one(label: str, stages: list[str], problem, params: dict):
    if stages == ["held_karp_exact"]:
        solver = get_algorithm("held_karp_exact")()
    elif len(stages) == 1:
        solver = get_algorithm(stages[0])()
    else:
        solver = make_pipeline(*stages, custom_name=label)
    return solver.solve(problem, **params)


init_state()

# Panel kiri: kontrol sesuai urutan alur kerja.
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
            help="Pilih beberapa metode. Metode konstruksi dapat digabung dengan pencarian lokal atau metaheuristik.",
        )
        st.caption("Pipeline dibuat dari metode konstruksi, lalu metode perbaikan yang dipilih.")

    with st.expander("3. Parameter", expanded=True):
        seed = st.number_input("Seed algoritma", min_value=0, max_value=999999, value=42,
                               help="Gunakan seed yang sama untuk mengulang eksperimen acak.")
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

# Main area: plain title, tabs for all visualizations.
st.title("Optimasi Rute TSP")

problem = st.session_state["problem"]
results = st.session_state["results"]

if problem is not None:
    st.caption(f"Data aktif: {st.session_state['problem_label']} · {problem.n} kota")
else:
    st.caption("Pilih data di panel kiri untuk memulai.")

if problem is not None and problem.n > 50 and "3-opt" in selected:
    st.warning("3-opt dapat berjalan lambat untuk lebih dari 50 kota. Pertimbangkan data yang lebih kecil.")
if problem is not None and problem.n > 18 and "Held-Karp (eksak)" in selected:
    st.warning("Held-Karp (eksak) dibatasi untuk maksimal 18 kota; metode ini tidak akan dijalankan.")

if run_button:
    if problem is None:
        st.error("Pilih atau muat data terlebih dahulu di bagian 1. Data.")
    elif not selected:
        st.error("Pilih minimal satu algoritma di bagian 2. Algoritma.")
    else:
        pipeline_specs = make_pipeline_names(selected)
        valid_specs = []
        for label, stages in pipeline_specs:
            if "held_karp_exact" in stages and problem.n > 18:
                continue
            valid_specs.append((label, stages))
        new_results = []
        params = {"seed": int(seed)}
        if sa_selected:
            params.update(T_max=float(t_max), T_min=float(t_min), alpha=float(alpha),
                          max_iter_per_temp=int(sa_iter))
        if tabu_selected:
            params.update(max_iterations=int(tabu_iterations), tabu_tenure=int(tabu_tenure),
                          neighborhood_size=int(tabu_neighborhood))
        progress = st.progress(0, text="Menyiapkan perhitungan")
        for i, (label, stages) in enumerate(valid_specs):
            progress.progress(i / max(len(valid_specs), 1), text=f"Menjalankan: {label}")
            try:
                sol = run_one(label, stages, problem, params)
                new_results.append({"label": label, "stages": stages, "solution": sol, "error": None})
            except Exception as exc:
                new_results.append({"label": label, "stages": stages, "solution": None, "error": str(exc)})
        progress.progress(1.0, text="Selesai")
        st.session_state["results"] = new_results
        st.session_state["last_run_key"] = (st.session_state["problem_label"], problem.n, int(seed))
        results = new_results
        if not valid_specs:
            st.warning("Tidak ada metode yang dapat dijalankan dengan batas ukuran data saat ini.")

tabs = st.tabs(["Rekomendasi", "Rute", "Konvergensi", "Perbandingan", "Detail"])
valid_results = [r for r in results if r.get("solution") is not None]
errors = [r for r in results if r.get("error")]

with tabs[0]:
    st.subheader("Rekomendasi algoritma")
    st.write(
        "Pilih metode sendiri seperti biasa. Bagian ini memberi saran awal berdasarkan jumlah kota "
        "dan karakteristik praktis algoritma; rekomendasi bukan jaminan solusi terbaik."
    )
    if problem is None:
        st.info("Muat data terlebih dahulu di panel kiri agar rekomendasi dapat disesuaikan dengan ukuran masalah.")
    else:
        st.caption(f"Analisis untuk {problem.n} kota.")
        for item in recommend_algorithms(problem):
            with st.container(border=True):
                st.caption(item["priority"])
                st.markdown(f"**{item['pipeline']}**")
                st.write(item["reason"])
                if item["pipeline"] == "Held-Karp (eksak)":
                    st.caption("Tersedia di aplikasi jika jumlah kota tidak lebih dari 18.")
        st.info(
            "Saran praktis: jalankan dua atau tiga kandidat dengan seed dan parameter yang sama, "
            "lalu pilih berdasarkan panjang rute dan waktu eksekusi pada tab Perbandingan."
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
            st.caption(f"Rute melewati {problem.n} kota, panjang {chosen_route['solution'].length:.2f}.")
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
        # Gunakan gaya garis dan marker berbeda, selain warna, untuk membedakan seri.
        ax = fig.axes[0]
        for line in ax.lines:
            if "Current" in line.get_label() or "current" in line.get_label().lower():
                line.set_color(CURRENT)
                line.set_linestyle("--")
                line.set_marker("x")
                line.set_markevery(max(1, len(line.get_xdata()) // 12))
                line.set_label("Rute yang sedang dicoba (garis putus-putus)")
            else:
                line.set_color(BEST)
                line.set_linestyle("-")
                line.set_marker("o")
                line.set_markevery(max(1, len(line.get_xdata()) // 12))
                line.set_label("Rute terbaik (garis penuh)")
        style_figure(fig)
        st.pyplot(fig, width="stretch")
        close_figure(fig)
        st.caption("Garis penuh menunjukkan rute terbaik sejauh ini; garis putus-putus menunjukkan rute yang sedang dicoba.")

with tabs[3]:
    if not valid_results:
        st.info("Jalankan setidaknya satu pipeline untuk melihat tabel perbandingan.")
    else:
        sorted_results = sorted(valid_results, key=lambda r: r["solution"].length)
        best_length = sorted_results[0]["solution"].length
        rows = []
        for rank, result in enumerate(sorted_results, 1):
            sol = result["solution"]
            insights = (sol.metadata or {}).get("insights", {})
            improvement = insights.get("improvement_pct")
            if improvement is None:
                improvement = 0.0
            rows.append({
                "Peringkat": rank,
                "Pipeline": result["label"],
                "Panjang rute": float(sol.length),
                "Waktu (detik)": float(sol.runtime_sec),
                "Perbaikan (%)": float(improvement),
                "Selisih dari terbaik (%)": ((sol.length - best_length) / best_length * 100) if best_length else 0.0,
            })
        comparison = pd.DataFrame(rows)
        st.dataframe(comparison, hide_index=True, width="stretch",
                     column_config={
                         "Panjang rute": st.column_config.NumberColumn(format="%.2f"),
                         "Waktu (detik)": st.column_config.NumberColumn(format="%.4f"),
                         "Perbaikan (%)": st.column_config.NumberColumn(format="%.2f"),
                         "Selisih dari terbaik (%)": st.column_config.NumberColumn(format="%.2f"),
                     })
        st.download_button("Unduh perbandingan CSV", comparison.to_csv(index=False).encode("utf-8"),
                           file_name="perbandingan_tsp.csv", mime="text/csv")
        st.caption(f"Tabel membandingkan {len(comparison)} pipeline; panjang rute terkecil adalah {best_length:.2f}.")

with tabs[4]:
    if not valid_results:
        st.info("Detail akan muncul setelah perhitungan selesai.")
    else:
        detail_choice = st.selectbox("Pilih pipeline untuk melihat detail",
                                     [r["label"] for r in valid_results],
                                     help="Pilih pipeline untuk melihat urutan kota dan ringkasan algoritma.")
        detail = next(r for r in valid_results if r["label"] == detail_choice)
        sol = detail["solution"]
        st.metric("Panjang rute", f"{sol.length:.2f}")
        st.metric("Waktu eksekusi", f"{sol.runtime_sec:.4f} detik")
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
