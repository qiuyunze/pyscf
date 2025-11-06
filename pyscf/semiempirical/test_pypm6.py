from pm6mole import PM6MOLE
from scf_diis import PM6HF
# from scf import PM6HF
from model_initialization import PM6Init

PM6env = PM6Init("./pm6_params.npz")
mol = PM6MOLE('./test/168_Valinomycin.xyz', PM6env, charge=0, spin=0, debug=False, verbose=6)
mf = PM6HF(mol)
ee, ehof = mf.kernel(diis=False)

print(ehof)