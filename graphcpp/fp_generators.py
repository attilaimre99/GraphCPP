from rdkit.Chem import rdFingerprintGenerator

fp_dict = {
    'morgan2': rdFingerprintGenerator.GetMorganGenerator(radius=2),
    'morgan3': rdFingerprintGenerator.GetMorganGenerator(radius=3),
    'morgan4': rdFingerprintGenerator.GetMorganGenerator(radius=4),
    'morgan8': rdFingerprintGenerator.GetMorganGenerator(radius=8),
    'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator(),
    'topological': rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True),
    'atompair': rdFingerprintGenerator.GetAtomPairGenerator(includeChirality=True),
}

