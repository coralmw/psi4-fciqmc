import numpy as np
import math
import random
from pprint import pprint
from functools import reduce, lru_cache, cache
from collections import Counter

import numpy as np
import psi4
from scipy.special import comb
from helper_CI import Determinant, HamiltonianGenerator
from itertools import combinations, product, count
import time

class DirectSolver():

    def __init__(self, mol):
        """Set up the QMC calc.
        """
        scf_e, wfn = psi4.energy('SCF', return_wfn=True)
        C = wfn.Ca()
        ndocc = wfn.doccpi()[0]
        nmo = wfn.nmo()
        
        self.M = nmo
        self.N = ndocc

        # Compute size of Hamiltonian in GB
        nDet = comb(nmo, ndocc)**2
        self.Ndet = int(nDet)
        H_Size = nDet**2 * 8e-9
        print('\nSize of the Hamiltonian Matrix would be %4.2f GB.' % H_Size)

        # Integral generation from Psi4's MintsHelper
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

        print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

        #Make spin-orbital MO
        print('Starting AO -> spin-orbital MO transformation...')
        t = time.time()
        MO = np.asarray(mints.mo_spin_eri(C, C))

        # Update H, transform to MO basis and tile for alpha/beta spin
        H = np.einsum('uj,vi,uv', C, C, H)
        H = np.repeat(H, 2, axis=0)
        H = np.repeat(H, 2, axis=1)

        # Make H block diagonal
        spin_ind = np.arange(H.shape[0], dtype=np.int64) % 2
        H *= (spin_ind.reshape(-1, 1) == spin_ind)

        print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

        # have to build this on the fly; should be fine
        self.HG = Hamiltonian_generator = HamiltonianGenerator(H, MO)
        
        def minWithIndex(idxValPair_l, idxValPair_r):
            if idxValPair_l[1] < idxValPair_r[1]:
                return idxValPair_l
            else:
                return idxValPair_r
        
        print('Finding min diagonal det')
        t = time.time()
        
        self.init_walker_idx, self.minHenergy = reduce(minWithIndex, map(lambda det: 
                                                 (det, self.HG.calcMatrixElement(det, det)), 
                                                 self.detBuilder()))
                                                      

        
        print(f"Found min diagonal det in {time.time()-t}s - idx {self.init_walker_idx} energy {self.minHenergy}")
        self.nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
        self.Ehf = self.minHenergy + self.nuclear_repulsion_energy
        self.population = Counter() # det-indexed walker counts.
        self.pop_history = []

        # FCI QMC evoluation hyperparameters. Comments give suggested ranges from the original paper.
        self.imagTau = 0.00001 # in the range 10−4 − 10−3 a.u
        self.damping = 0.1 #  0.05— 0.1
        self.A = 10# 5 - 10
        self.S = 0.0
        self.startup = self.A * 5

        self.population[self.init_walker_idx] = 5
        self.iteration = 0
        
        # probabilty scaling factor for self interactions. Should sum to unity; this is from
        # the condmat presentation but there is a alternate definition in the paper.
        self.scalingFactor = 1.0/(self.N**2 * self.M**2 + self.N*self.M)


    def detBuilder(self):
        """To avoid storing the exponential number of determinants in memory,
        this helper function returns a generator object to iterate over all dets.
        This is only used to find the single lowest energy det now, and even this is not required.
        
        the apha and beta opbital occs are just 2 ndet-bit numbers; this could be random access as well.
        """
        return map(lambda detIdx: Determinant(alphaObtList=detIdx[0], betaObtList=detIdx[1]), 
                   product(combinations(range(self.M), self.N), repeat=2))


    def C_i(self, i):
        return self.population[i]

    def N_w(self):
        return sum(map(lambda kv: abs(kv[1]), self.population.items()))
    
    def energy(self):
        i = self.init_walker_idx
        if self.population[i] == 0.0:
            return float('NaN')
        return self.Ehf + sum( [ self.Hij(i, j) * self.population[j]/self.population[i] 
                                 for j in self.detBuilder() if j != i] )
    
    def Hij(self, i, j):
        "we call Hij O(n) rather than N^2 times per iter, so better to cache the K version."
        return self.Kij(i, j) + (self.Ehf if i==j else 0.0)
    
    @lru_cache(2**22) # 2 32b hash keys + 64b value. We hope 
    def Kij(self, i, j):
        return self.HG.calcMatrixElement(i, j) - (self.Ehf if i==j else 0.0)

    @staticmethod
    def update_count(count, probabilty, threshold=7):
        """
        Every iteration involves a loop over each walker in a determinant.
        Most occupied dets will have many walkers, so the random process needs to 
        generate an effect from each walker with identical, uniform, chance.
        
        A sum of IID uniform is given by the Irwin-Hall distribution; this is not
        provided. OTOH by central limit IH rapidly converges to normal. 
        
        This function returns the number of IID uniform draws that are > probabilty
        out of count draws; if count > threshold this is calculated in linear time
        with a normal approximation. The default threshold is set to "close by eye"
        """
        if count < threshold:
            update = 0
            for walker in range(count):
                update += math.floor(probabilty)
                if probabilty - math.floor(probabilty) > random.uniform(0,1):
                    update += 1
        else: # central limit theorem
            update = np.random.normal(loc=count * probabilty, scale=np.sqrt(count/6), size=None)
        return int(update)


    def spawn(self):
        update = Counter()
        for i_ao in self.population.keys():
            for j_ao in i_ao.generateSingleAndDoubleExcitationsOfDet(self.M):
                count = abs(self.population[i_ao])
                if i_ao == j_ao:
                    continue
                    
                if not i_ao.diff2OrLessOrbitals(j_ao):
                    continue
                
                kij = self.Kij(i_ao, j_ao)
                if abs(kij) < 1e-10:
                    continue
                
                P_spawn_j_given_i = self.imagTau * abs(kij) / self.scalingFactor
                if P_spawn_j_given_i < 1e-10:
                    continue

                if kij < 0:
                    sign_child = int(math.copysign(1, self.population[i_ao]))
                else:
                    sign_child = int(math.copysign(1, self.population[i_ao]) * -1)
                    
                events = self.update_count(count, P_spawn_j_given_i)
                if events != 0:
                    update[j_ao] += int(sign_child) * events
                
        return update

    def d_c(self):
        update = Counter()
        for i_ao in self.population.keys():
            count = abs(self.population[i_ao])
            P_d = self.imagTau * (self.Kij(i_ao, i_ao) - self.S)
            events =  self.update_count(count, abs(P_d))
            if events == 0:
                continue
            if P_d > 0:
                # death step; dec the walker count on i_ao by 1 absolute value
                update[i_ao] += -1 * int(math.copysign(1, self.population[i_ao])) * events
            if P_d < 0:
                # cloning step; incr by 1 abs value.
                update[i_ao] += int(math.copysign(1, self.population[i_ao])) * events
        return update

    # def annihilation(self):
    #     update = np.zeros(self.Ndet, dtype=int)
    #     # because we have the n^2 list of dets, the annihilation step is free.
    #     return update

    def adjust_shift(self):
        self.S = self.S - (self.damping / (self.A*self.imagTau)) * \
                           np.log( self.pop_history[-1]/self.pop_history[-self.A] )

    def step(self):
        count = self.N_w()
        if count == 0:
            raise RuntimeError(f"tried to take a MC step with zero population; iter {len(self.pop_history)}")
        self.pop_history.append(count)

        s_update = self.spawn()
        d_update = self.d_c()
        self.population.update(s_update)
        self.population.update(d_update)
        # ann_update = self.annihilation()
        # self.population += ann_update
        self.iteration += 1

        if self.iteration % self.A == 0 and self.iteration > self.startup:
            self.adjust_shift()
        return (s_update, d_update, None)

def main():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """)
    
    # MOL most have c1 (no) symmetry; this is assumed for building parts of the FCI calculation.
    # mol = psi4.geometry("""
    # H
    # H 1 1.1
    # symmetry c1
    # """)
    
    # mol = psi4.geometry("""
    # Be
    # Be 1 2.45
    # symmetry c1
    # """)


    
    psi4.set_options({'basis': 'sto-3g',
                      'scf_type': 'pk',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})

    print('\nStarting SCF and integral build...')
    t = time.time()

    # First compute SCF energy using Psi4
    scf_e = psi4.energy('SCF')
    E_FCI = psi4.energy('CCSD')
    
    s = DirectSolver(mol)
    # run the real script
    N = 1000
    energy_samples = np.zeros(N)
    population_counts = np.zeros(N)
    delta_spawn = np.zeros(N)
    delta_death_clone = np.zeros(N)
    shifts = np.zeros(N)
    ts = time.time()
    iter_s = 0
    for i in range(N):
        dt = time.time() - ts
        diter = i - iter_s
        if dt>5:
            print(f"iter {i} pop {s.N_w()} shift {s.S} energy {s.energy()} iter/sec {diter/dt}")
            ts = time.time()
            iter_s = i
        s_update, d_update, ann_update = s.step()
        delta_spawn[i] = sum(map(lambda kv: abs(kv[1]), s_update.items()))
        delta_death_clone[i] = sum(map(lambda kv: abs(kv[1]), d_update.items()))
        energy_samples[i] = s.energy()
        population_counts[i] = s.N_w()
        shifts[i] = s.S
    # print("final state/population count", s.population)
    E_est = np.mean(energy_samples[~np.isnan(energy_samples)])
    print(f"final energy estimate: {E_est} error: {abs(E_est-E_FCI)}")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True)
    axs[0].plot(population_counts, label="population")
    axs[0].plot(delta_spawn, label="delta_spawn")
    axs[0].plot(delta_death_clone, label="delta_death_clone")
    axs[0].set_yscale("log")
    axs[1].plot(shifts, label="shift")
    axs[2].plot(np.abs(energy_samples-E_FCI), label="energy")
    axs[2].set_yscale("log")
    
    for ax in axs:
        ax.legend()
    import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     fig.show()
    plt.show()

if __name__ == "__main__":
    main()