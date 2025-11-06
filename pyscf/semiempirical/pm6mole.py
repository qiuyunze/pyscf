from pyscf import scf
import sys, time
import numpy as np
from pathlib import Path
from model_initialization import PM6Init
from pyscf.gto.mole import format_atom
from pyscf.data.elements import ELEMENTS
from mndod import calpar
from mndod import rotate_np as rotate
from mndod import wstore_np as wstore
from h1elec import h1elec_np as h1elec
from utils import print_title, vecprt_w, vecprt_h, _pack_index, prtpar, packed_to_full, full_to_packed
from fock import fock2_np as fock2
from atheat import compute_atheat_pm6_mol
from pyscf.lib import logger

class PM6MOLE:
    def __init__(self, atoms: list or str, PM6env: PM6Init, charge: int = 0, spin: int = 0, id_dim: int = 0, verbose=0,debug: bool = False):
        self.atoms = atoms
        self.PM6env = PM6env
        self.charge = charge
        self.spin = spin
        self.id_dim = id_dim
        self.verbose = verbose
        log = logger.new_logger(self, self.verbose)

        fa = format_atom(atoms, unit='Bohr')
        Z = np.asarray([ELEMENTS.index(atom[0]) for atom in fa], dtype=np.int32)
        coord = np.asarray([atom[1] for atom in fa], dtype=np.float64).T
        self.Z = Z
        self.coord = coord
        self.numat = int(Z.size)
        #  electron number
        self.nelecs = sum([self.PM6env.tore[z-1] for z in Z]) - int(self.charge)
        self.nalpha, self.nbeta, self.rhf = self._nelecs_to_nalpha_nbeta()
        self.nclose = self.nelecs // 2 if self.rhf else 0
        self.nopen  = (self.nelecs - 2*self.nclose) if self.rhf else 0
        # orbital number
        self.natorb = np.array([self._natorb_for_Z(int(z)) for z in Z], dtype=int)
        self.nfirst, self.nlast, self.norbs = self._nfirst_nlast_norbs()
        self.mpack = self.norbs * (self.norbs + 1) // 2
        # diagnol term
        self.uspd = self._uspd_init()
        # initial guess
        self.pdiag = self._pdiag_init()
        # 2-electron integrals guess number
        self.n2elec = self._n2elec_guess()
        # required arrays
        npulay = 3
        if id_dim == 0:
            l123 = 1  # QYZ: 1 for molecular calculations
        else:
            raise ValueError("Id_dim must be 0 for molecular calculations; periodic calculations are not supported yet.")
        zeros_vec_mpack = np.zeros(self.mpack, dtype=np.float64)
        zeros_mat_nn = np.zeros((self.norbs, self.norbs), dtype=np.float64)
        zeros_pold = np.zeros(npulay * self.mpack, dtype=np.float64)
        self.grad = np.zeros(self.numat *3, dtype=np.float64)
        # one-electron integrals and density matrix
        self.h = zeros_vec_mpack.copy()
        # self.hfull = zeros_mat_nn.copy()  
        self.p = zeros_vec_mpack.copy()
        # self.pfull = zeros_mat_nn.copy()
        self.pa = zeros_vec_mpack.copy()
        self.pb = zeros_vec_mpack.copy()
        # pulay history
        self.pold = zeros_pold.copy()
        self.pold2 = zeros_pold.copy()
        self.pold3 = np.zeros(max(self.mpack, 400), dtype=np.float64)
        # Fock matrix
        self.f = zeros_vec_mpack.copy()
        # self.ffull = zeros_mat_nn.copy()
        self.c = zeros_mat_nn.copy()
        self.M = zeros_mat_nn.copy()    # which would be used in the SCF loop
        self.eigs = np.zeros(self.norbs, dtype=np.float64)
        self.eigb = np.zeros(self.norbs, dtype=np.float64)
        # two-electron integrals
        self.w = np.zeros(self.n2elec + 2025, dtype=np.float64)
        if l123 > 1:
            self.wk = np.zeros(self.n2elec + 2025, dtype=np.float64)
        else:
            self.wk = self.w   # share the same memory as w
        self.dxyz = np.zeros(3*self.numat*l123, dtype=np.float64)
        if not self.rhf:
            self.fb = zeros_vec_mpack.copy()
            self.cb = zeros_mat_nn.copy()
            self.pbold = zeros_pold.copy()
            self.pbold2 = zeros_pold.copy() 
            self.pbold3 = np.zeros(max(self.mpack, 400), dtype=np.float64)  # remain to be checked
        # in-situ update the PM6 environment
        calpar(env=self.PM6env)
        if debug:
            prtpar(self, self.PM6env)
        # calculate the core Hamiltonian including 1-electron integrals and 2-electron integrals
        t0 = time.time()
        self._hcore(PM6env=self.PM6env, debug=debug)   # QYZ: refine the code 
        t1 = time.time()
        log.debug(f"Hcore time: {t1 - t0:.6f} s")
        self.atheat = compute_atheat_pm6_mol(self.Z, self.coord, eheat_lookup=self.PM6env.eheat, eisol_lookup=self.PM6env.eisol)['atheat']
        self.fract = 0.0
        self._interface_to_pyscf()

    def _nelecs_to_nalpha_nbeta(self) -> tuple:
        spin = self.spin
        if spin is None:
            if self.nelecs % 2 == 0:
                spin = 0
            else:
                spin = 1
        nalpha = (self.nelecs + self.spin) // 2
        nbeta  = self.nelecs - nalpha
        rhf = True
        if spin != 0:
            rhf = False
        return nalpha, nbeta, rhf

    def _natorb_for_Z(self, z: int) -> int:
        thr_d = 1e-12; thr_p = 1e-20; thr_s = 1e-20
        i = z - 1
        zs6 = self.PM6env.zs6
        zd6 = self.PM6env.zd6
        zp6 = self.PM6env.zp6
        if i < 0 or i >= zs6.size:
            return 0
        if zd6[i] > thr_d:        return 9
        if zp6[i] > thr_p:        return 4
        if zs6[i] > thr_s:        return 1
        return 0    

    def _nfirst_nlast_norbs(self) -> tuple:
        nfirst = np.zeros(self.numat, dtype=int)
        nlast  = np.zeros(self.numat, dtype=int)
        ia = 0
        for i in range(self.numat):
            k = int(self.natorb[i])
            if k == 0:
                nfirst[i] = ia
                nlast[i]  = ia - 1
            else:
                nfirst[i] = ia
                nlast[i]  = ia + k - 1
                ia += k
        return nfirst, nlast, int(ia)

    def _uspd_init(self) -> np.ndarray:
        uspd = np.zeros(self.norbs, dtype=np.float64)
        uss6 = self.PM6env.uss6
        upp6 = self.PM6env.upp6
        udd6 = self.PM6env.udd6
        for i in range(self.numat):
            a, b = int(self.nfirst[i]), int(self.nlast[i])
            if b < a:    # without AO
                continue
            zi = int(self.Z[i]) - 1
            n_orb = b - a + 1
            if n_orb == 1:
                uspd[a] = uss6[zi]
            elif n_orb == 4:
                uspd[a]     = uss6[zi]
                uspd[a+1:a+4] = upp6[zi]
            elif n_orb == 9:
                uspd[a]       = uss6[zi]
                uspd[a+1:a+4] = upp6[zi]
                uspd[a+4:a+9] = udd6[zi]
            else:
                raise RuntimeError(f"Incorrect number of orbitals for atom, which should be 1/4/9, but got {n_orb}")
        return uspd

    def _pdiag_init(self) -> np.ndarray:
        '''
        Initialize the pdiag vector; important for smooth SCF convergence
        '''
        pdiag = np.zeros(self.norbs, dtype=int)
        
        charge = self.charge
        nfirst, nlast, norbs = self.nfirst, self.nlast, self.norbs
        yy = float(charge) / (self.norbs + 1e-10)
        pdiag = np.zeros(norbs, dtype=float)
        Z = self.Z
        tore = self.PM6env.tore
        for i in range(self.numat):
            if nlast[i] - nfirst[i] == -1:
                continue
            l0 = int(nfirst[i]) 
            b  = int(nlast[i])  
            n_orb = b - l0 + 1
            zi = int(Z[i])-1    # numpy 0-base
            te = float(tore[zi])

            if n_orb == 1:
                # Hydrogen-like
                pdiag[l0] = te - yy

            elif n_orb == 4:
                # Normal heavy atom (s+p only)
                w = 0.25 * te - yy
                pdiag[l0:l0+4] = w

            else:
                # d shell
                if (zi < 21) or (30 < zi < 39) or (48 < zi < 57):
                    w = 0.25 * te - yy
                    pdiag[l0:l0+4] = w            # s+p
                    pdiag[l0+4:l0+9] = -yy         # 5 d orbitals
                elif zi < 99:
                    sum_e = te - 9.0 * yy

                    s_occ = max(0.0, min(sum_e, 2.0))
                    pdiag[l0] = s_occ
                    sum_e -= s_occ

                    if sum_e > 0.0:
                        d_vals = []
                        for _ in range(5):
                            occ = max(0.0, min(0.2 * sum_e, 2.0))
                            d_vals.append(occ)
                        d_vals = np.array(d_vals)
                        pdiag[l0+4:l0+9] = d_vals
                        sum_e -= 10.0  

                        if sum_e > 0.0:
                            p_occ = sum_e / 3.0
                            pdiag[l0+1:l0+4] = p_occ
                        else:
                            pdiag[l0+1:l0+4] = 0.0
                    else:
                        pdiag[l0+1:l0+4] = 0.0
                        pdiag[l0+4:l0+9] = 0.0
                else:
                    pdiag[l0:l0+n_orb] = 0.0
        return pdiag
    
    def _n2elec_guess(self) -> int:
        has1 = int(np.sum(self.natorb == 1))
        has4 = int(np.sum(self.natorb == 4))
        has9 = int(np.sum(self.natorb == 9))
        ispd = int(has9)
        if self.id_dim == 0:   # molecule
            n2 = (has4*(has4 - 1))//2
            n2 = 100*n2 + 2025*has9 + 100*has4 + has1 \
                + 2025*((has9*(has9 - 1))//2) + 450*has9*has4 + 45*has9*has1 \
                + 10*has4*has1 + (has1*(has1 - 1))//2 + 10
        else:         # periodic system    
            n2 = (has4*(has4 + 1))//2
            n2 = 100*n2 + 2025*has9 + 100*has4 + has1 \
                + 2025*((has9*(has9 + 1))//2) + 450*has9*has4 + 45*has9*has1 \
                + 10*has4*has1 + (has1*(has1 + 1))//2 + 10
        n2elec = int(min(n2, 2147483647))
        return n2elec
    
    def _hcore(self, PM6env, debug=False):
        numat  = self.numat
        norbs  = self.norbs
        mpack  = self.mpack

        nfirst = np.asarray(self.nfirst, dtype=int)  
        nlast  = np.asarray(self.nlast,  dtype=int)   
        nat    = np.asarray(self.Z,    dtype=int)   
        uspd   = np.asarray(self.uspd,   dtype=float) 
        coord  = np.asarray(self.coord,  dtype=float) 

        h = self.h   # packed one-electron integrals, length = mpack
        w = self.w   # two-electron integrals 
        iw = sys.stdout
        # ---- initialize the h matrix ----
        enuclr = 0.0
        kr = 0  # current position in the w matrix
        def _add_packed_block(h, ia, ib, eblk, scale=1.0):
            """
                Add the packed block eblk (length L*(L+1)//2) to the h matrix.
                The block is added to the global row p=ia+r in the column interval [ia, ia+r].
            """
            L = ib - ia + 1
            if L <= 0:
                return
            off = 0
            for p in range(ia, ib + 1):
                n = p - ia + 1
                row_vals = eblk[off:off + n]
                # add the row_vals to the global h matrix
                base = _pack_index(p, ia)
                h[base: base + n] += scale * row_vals
                off += n

        # ====== Atom Pair Loop ======
        ### For each atom pair, the number of two-electron integrals is less than.
        for i in range(numat):
            ia = int(nfirst[i])
            ib = int(nlast[i])
            ni = int(nat[i])                

            # 1) fill the diagonal of the matrix one-electron integrals with the uspd values
            for p in range(ia, ib + 1):
                h[_pack_index(p, p)] = float(uspd[p])

            # 2) off-diagonal matrix one-electron integrals and two-electron integrals
            for j in range(0, i):
                ja = int(nfirst[j]); jb = int(nlast[j]); nj = int(nat[j])

                # 2a) fill the off-diagonal of the matrix one-electron integrals with the h1elec values
                di = np.asarray(h1elec(ni, nj, coord[:, i], coord[:, j], env=PM6env), dtype=float)   # Note: atomic number is the input, not 0-based
                # di's shape : ((ib-ia+1),(jb-ja+1))
                for p in range(ia, ib + 1):
                    row = p - ia
                    max_col = min(p, jb)
                    if max_col >= ja:
                        base = _pack_index(p, ja)
                        n = max_col - ja + 1
                        h[base: base + n] += di[row, :n]
        
                # 2b) get the two-electron integrals & nuclear repulsion energy from the rotate function
                kr_new, e1b, e2a, enuc_add = rotate(ni, nj, coord[:, i], coord[:, j], w, kr, PM6env)
                kr = int(kr_new)
                enuclr += float(enuc_add)

                # 2c) add the e1b to the global h matrix
                _add_packed_block(h, ia, ib, np.asarray(e1b, dtype=float), scale=1.0)
                # 2d) add the e2a to the global h matrix
                _add_packed_block(h, ja, jb, np.asarray(e2a, dtype=float), scale=1.0)

            # 3) one-center two-electron integrals
            L = ib - ia + 1
            ilim = (L * (L + 1)) // 2
            if wstore is not None and ilim > 0:
                kr = int(wstore(w, kr, ni, ilim, PM6env))
        if kr > 0:
            self.w = w[:kr]
        
  
        self.enuc = float(enuclr)   # hcore core-core repulsion（eV）   
        self.kr     = int(kr)
        if debug:
            # print_title(iw, "ONE-ELECTRON MATRIX FROM HCORE", leading_blank_lines=2, trailing_blank_lines=1)
            # vecprt_h(self)
            vecprt_w(iw, self.h, "ONE-ELECTRON MATRIX IN HCORE")
            vecprt_w(iw, self.w, "TWO-ELECTRON MATRIX IN HCORE")

    def _interface_to_pyscf(self):
        # === minimum PySCF interface ===
        self._built = True                 # used in pyscf/scf/hf.py
        self.max_memory = 2000
        self.stdout = sys.stdout           
        self.nao = int(self.norbs)         # AO
        self.norb = self.nao               
        self.nelectron = int(self.nelecs)  
        self.nelec = (int(self.nalpha), int(self.nbeta))  
        self.spin = int(self.nalpha - self.nbeta)        
        self.charge = int(self.charge)     

        def build(*args, **kwargs):        # PySCF call mol.build() 
            self._built = True
            return self
        self.build = build

        def nao_nr():
            return self.nao
        self.nao_nr = nao_nr