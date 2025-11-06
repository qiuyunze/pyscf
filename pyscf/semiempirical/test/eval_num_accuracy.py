from ase.build import molecule
from ase import Atoms
from ase.io import read
from ase.calculators.mopac import MOPAC
import numpy as np 
import sys
sys.path.append('/mlx_devbox/users/qiuyunze.458/playground/package/pySQC/package_6')
from pm6mole import PM6MOLE
from scf_diis import PM6HF
from model_initialization import PM6Init
import sys
import time
import glob

from pathlib import Path

xyz_files = sorted(Path('.').glob('*.xyz'))
if not xyz_files:
    raise FileNotFoundError("当前目录下没有 .xyz 文件")
elif len(xyz_files) == 1:
    atoms = read(str(xyz_files[0]))
else:
    latest = max(xyz_files, key=lambda p: p.stat().st_mtime)
    atoms = read(str(latest))
label = str(xyz_files[0]).split('.')[0]

start = time.perf_counter()
atoms.calc = MOPAC(label=label, 
                   task='1SCF HCORE RELSCF=0.0001 GRADIENTS PREC FORCE',
                   method='PM6', 
                   command='/mlx_devbox/users/qiuyunze.458/playground/github/mopac2/mopac/build/mopac PREFIX.mop > log.txt')
atoms.get_potential_energy()
end = time.perf_counter()
print(f"MOPAC 耗时：{end - start:.6f} 秒")


start = time.perf_counter()
PM6env = PM6Init("/mlx_devbox/users/qiuyunze.458/playground/package/pySQC/package1/pm6_params.npz")
mol = PM6MOLE(str(xyz_files[0]), PM6env, charge=0, spin=0, debug=False, verbose=4)
mf = PM6HF(mol) 
etot, ehof = mf.kernel(diis=True)   # Fortran energy reference: TOT: -12344.91087; HOF： -187.47852 
end_energy = time.perf_counter()
grad = mf.pm6_dcart_gradient(prec=True)
end = time.perf_counter()
print(f"PYTHON-PM6 Energy 耗时：{end_energy - start:.6f} 秒")
print(f"PYTHON-PM6 Gradient 耗时：{end - end_energy:.6f} 秒")


def read_mat_element(file):
    fock = []
    with open(file, 'r') as f:
        lines = f.readlines()
        h = []
        w = []
        for idx, line in enumerate(lines):
            if 'ONE-ELECTRON MATRIX FROM HCORE' in line:
                idx1 = idx
            if 'TWO-ELECTRON MATRIX IN HCORE' in line:
                idx2 = idx
            if 'SCF CRITERION' in line:
                idx3 = idx
            if 'TOTAL ENERGY            = ' in line and 'EV' in line:
                etot_fortran = float(line.split()[-2])
        seg1 = lines[idx1+1:idx2]
        seg2 = lines[idx2+1:idx3]

        for line in seg1:
            for val in line.split():
                h.append(float(val))
        for line in seg2:
            for val in line.split():
                w.append(float(val))
    return np.array(h), np.array(w), etot_fortran
def read_gradient(file):
    grad = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'DOING VARIATIONALLY OPTIMIZED DERIVATIVES' in line:
                idx1 = idx
            if 'MOPAC Job' in line:
                idx2 = idx
        seg1 = lines[idx1+1:idx2]  
        for line in seg1: 
            for val in line.split():
                grad.append(float(val))
    return np.array(grad)

h_fortran, w_fortran, etot_fortran = read_mat_element(label+'.out')
grad_fortran = read_gradient('log_gradient_wtihout_MM_force').reshape(-1, 3)

Hartree2eV = 27.211386245988
EV2KCALMOL = 23.060547830619029 
print('Largest deviation in one-electron integrals: {:e} Ha. '.format(np.max(abs(h_fortran - mol.h))/Hartree2eV))
print('Largest deviation in two-electron integrals: {:e} Ha. '.format(np.max(abs(w_fortran - mol.w))/Hartree2eV))
print('Deviation in total energy: {:e} Ha. '.format(abs(etot - etot_fortran)/Hartree2eV))
print('Largest deviation in gradient: {:e}  Ha./Angs.'.format(np.max(abs(grad_fortran - grad))/(Hartree2eV*EV2KCALMOL)))
import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.title(label)
plt.scatter(range(len(h_fortran)), (h_fortran - mol.h)/Hartree2eV)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=False)
plt.ylabel('Dev. of one-electron int. (Ha.)')

plt.subplot(2,1,2)
plt.scatter(range(len(w_fortran)), (w_fortran - mol.w)/Hartree2eV, c='orange')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=False)
plt.xlabel('Index')
plt.ylabel('Dev. of two-electron int. (Ha.)')

plt.savefig(label+'_num_accuracy.png', format='png', bbox_inches='tight')
