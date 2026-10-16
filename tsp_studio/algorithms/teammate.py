"""Algoritma kontribusi anggota tim (versi dasar).

Logika asli dipertahankan. Perubahan dari file aslinya hanya:
  * nama kelas dan nama registry diberi akhiran agar tidak bentrok dengan versi utama
  * import NearestNeighbor di ThreeOptSwap diperbaiki (`nearest_neighborhood` ->
    `nearest_neighbor`), supaya tidak error saat initial_tour tidak diberikan

Isi:
  farthest_insertion_far_start  farthest insertion, tour awal = kota awal + kota terjauhnya
  three_opt_swap                local search: tukar dua segmen (satu jenis move 3-opt)
  simulated_annealing_basic     SA sederhana (hitung ulang panjang tour penuh per tetangga)
  tabu_search_basic             Tabu Search sederhana (tetangga 2-opt acak)

Parameter berbeda dari versi utama: t_max, t_min, alpha, iter_per_temp (SA) dan
max_iter, tabu_tenure, neighborhood_size (Tabu). Keduanya memakai random global,
jadi parameter `seed` diabaikan (hasil tidak reproducible).
"""
from __future__ import annotations
import math
import random
import time
from typing import Optional

from ..core.problem import TSPProblem
from .base import Solver, Solution


class FarthestInsertionFarStart(Solver):
    name = "farthest_insertion_far_start"

    def solve(self, problem: TSPProblem, *, start: int = 0, **_) -> Solution:
        t0 = time.perf_counter()
        n = problem.n
        if not 0 <= start < n:
            raise ValueError(f"start must be in [0,{n})")

        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]

        if unvisited:
            nxt = max(unvisited, key=lambda j: problem.dist(start, j))
            tour.append(nxt)
            unvisited.remove(nxt)

        while unvisited:
            # Cari node terjauh dari tour
            u = max(unvisited, key=lambda v: min(problem.dist(v, node) for node in tour))
            unvisited.remove(u)
            
            # Cari posisi sisip terbaik
            best_cost = float('inf')
            best_idx = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                cost = problem.dist(tour[i], u) + problem.dist(u, tour[j]) - problem.dist(tour[i], tour[j])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i + 1
                    
            tour.insert(best_idx, u)

        return self._make_solution(problem, tour, time.perf_counter() - t0, start=start)


class ThreeOptSwap(Solver):
    name = "three_opt_swap"

    def solve(self, problem: TSPProblem, *, initial_tour: Optional[list[int]] = None, max_passes: Optional[int] = None, **_) -> Solution:
        t0 = time.perf_counter()
        
        if initial_tour is None:
            # Gunakan Nearest Neighbor sebagai baseline jika tidak ada rute awal
            from .nearest_neighbor import NearestNeighbor
            initial_tour = NearestNeighbor().solve(problem).tour
        else:
            problem.validate_tour(initial_tour)
            initial_tour = list(initial_tour)

        n = len(initial_tour)
        tour = initial_tour
        mat = problem.distance_matrix
        
        if n < 6:
            return self._make_solution(problem, tour, time.perf_counter() - t0)

        improved = True
        passes = 0
        
        while improved:
            improved = False
            passes += 1
            
            for i in range(n - 2):
                a, b = tour[i], tour[i + 1]
                for j in range(i + 2, n - 1):
                    c, d = tour[j], tour[j + 1]
                    for k in range(j + 2, n):
                        e, f = tour[k], tour[(k + 1) % n]
                        
                        base_cost = mat[a][b] + mat[c][d] + mat[e][f]
                        # 3-opt move variant: A-D-E-B-C-F
                        new_cost = mat[a][d] + mat[e][b] + mat[c][f]
                        
                        if new_cost < base_cost - 1e-9:
                            tour = tour[:i+1] + tour[j+1:k+1] + tour[i+1:j+1] + tour[k+1:]
                            improved = True
                            break
                    if improved: break
                if improved: break
                
            if max_passes is not None and passes >= max_passes:
                break

        return self._make_solution(problem, tour, time.perf_counter() - t0, max_passes=max_passes)


class SimulatedAnnealingBasic(Solver):
    name = "simulated_annealing_basic"

    def solve(self, problem: TSPProblem, *, initial_tour: Optional[list[int]] = None, 
              t_max: float = 1000.0, t_min: float = 0.001, alpha: float = 0.995, 
              iter_per_temp: int = 100, **_) -> Solution:
        t0 = time.perf_counter()
        
        if initial_tour is None:
            # Buat rute acak jika tidak ada rute awal
            current_tour = list(range(problem.n))
            random.shuffle(current_tour)
        else:
            problem.validate_tour(initial_tour)
            current_tour = list(initial_tour)

        current_dist = problem.tour_length(current_tour)
        best_tour = current_tour.copy()
        best_dist = current_dist
        
        T = t_max
        history = []
        n = problem.n
        
        while T > t_min:
            for _ in range(iter_per_temp):
                # 2-opt random neighbor
                i, j = sorted(random.sample(range(n), 2))
                neighbor_tour = current_tour.copy()
                neighbor_tour[i+1:j+1] = reversed(current_tour[i+1:j+1])
                
                neighbor_dist = problem.tour_length(neighbor_tour)
                delta = neighbor_dist - current_dist
                
                # Metropolis criterion
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_tour = neighbor_tour
                    current_dist = neighbor_dist
                    
                    if current_dist < best_dist:
                        best_tour = current_tour.copy()
                        best_dist = current_dist
                        
            history.append(best_dist)
            T *= alpha
            
        return self._make_solution(problem, best_tour, time.perf_counter() - t0, 
                                   t_max=t_max, t_min=t_min, alpha=alpha, history=history)


class TabuSearchBasic(Solver):
    name = "tabu_search_basic"

    def solve(self, problem: TSPProblem, *, initial_tour: Optional[list[int]] = None, 
              max_iter: int = 300, tabu_tenure: int = 20, neighborhood_size: int = 50, **_) -> Solution:
        t0 = time.perf_counter()
        
        if initial_tour is None:
            current_tour = list(range(problem.n))
            random.shuffle(current_tour)
        else:
            problem.validate_tour(initial_tour)
            current_tour = list(initial_tour)

        current_dist = problem.tour_length(current_tour)
        best_tour = current_tour.copy()
        best_dist = current_dist
        
        tabu_list = {}
        history = []
        n = problem.n
        
        for iteration in range(max_iter):
            best_neighbor = None
            best_neighbor_dist = float('inf')
            best_move = None
            
            for _ in range(neighborhood_size):
                i, j = sorted(random.sample(range(n), 2))
                move = (i, j)
                neighbor = current_tour.copy()
                neighbor[i+1:j+1] = reversed(current_tour[i+1:j+1])
                neighbor_dist = problem.tour_length(neighbor)
                
                is_tabu = tabu_list.get(move, 0) > iteration
                aspiration = neighbor_dist < best_dist
                
                if (not is_tabu) or aspiration:
                    if neighbor_dist < best_neighbor_dist:
                        best_neighbor = neighbor
                        best_neighbor_dist = neighbor_dist
                        best_move = move
                        
            if best_neighbor is not None:
                current_tour = best_neighbor
                current_dist = best_neighbor_dist
                tabu_list[best_move] = iteration + tabu_tenure
                
                if current_dist < best_dist:
                    best_tour = current_tour.copy()
                    best_dist = current_dist
                    
            history.append(best_dist)
            
        return self._make_solution(problem, best_tour, time.perf_counter() - t0, 
                                   max_iter=max_iter, tabu_tenure=tabu_tenure, history=history)
