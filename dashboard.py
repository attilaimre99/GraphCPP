import streamlit as st
import pandas as pd
import torch
import uuid
import math
from graphcpp.lightning import GraphCPPModule
from graphcpp.dataset import _featurize_mol
from graphcpp.fp_generators import fp_dict
from config import AVAIL_GPUS
from rdkit import Chem
from io import StringIO
import yaml

# ----------- We load the model once.
# We load in the best model from the model path
with open('model/hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
model = GraphCPPModule.load_from_checkpoint(checkpoint_path="model/checkpoints/epoch=50-step=1173.ckpt", map_location=None if AVAIL_GPUS>0 else torch.device('cpu'))
model.eval()
model.freeze()

# ----------- Default values
submit = None
predictions = list()

# ----------- General things
st.header('GraphCPP: The new state-of-the-art method for cellpenetrating peptide prediction via graph neural networks')
st.caption("[Attila Imre](https://orcid.org/0009-0004-0110-8395) <sup>1,2,3</sup>, [Balázs Balogh PhD](https://orcid.org/0000-0001-7282-7283) <sup>1,3</sup>, [István Mándity PhD](https://orcid.org/0000-0003-2865-6143) <sup>1,3,4</sup>", unsafe_allow_html=True)

st.markdown("""
<sup>1</sup> *Department of Organic Chemistry, Faculty of Pharmacy, Semmelweis University, Budapest, Hungary*\n
<sup>2</sup> *Center for Health Technology Assessment, Semmelweis University, Budapest, Hungary*\n
<sup>3</sup> *Center for Pharmacology and Drug Research & Development, Semmelweis University, Budapest, Hungary*
<sup>4</sup> *Artificial Transporters Research Group, Research Centre for Natural Sciences, Budapest, Hungary*
""", unsafe_allow_html=True)

# ----------- Input fields
smiles_strings = st.text_area('Prediction on **SMILES** in **csv** format. If name is not supplied a random name will be generated. To allow access to everyone in the research community this server is limited to 100 samples per run.', help='Enter valid SMILES strings.', placeholder='name,smiles\nfirst,CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCSC)NC(=O)[C@H](CS)NC(=O)[C@H](CCCCN)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]cn1)NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@@H](N)CCCCN)[C@@H](C)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O')
fasta_strings = st.text_area('Prediction on **FASTA**. If name is not supplied a random name will be generated. To allow access to everyone in the research community this server is limited to 100 samples per run.', help='Enter valid FASTA.', placeholder='>First\nRRRRRRRR\n>Second\nWWWWWWWW')

# ----------- SMILES processing
smis = pd.DataFrame()
if len(smiles_strings) > 0:
    # we read in the text as it was a csv file without header
    smis = pd.read_csv(StringIO(smiles_strings))
    # we generate uuid4 names for empty name fields
    smis['name'] = smis['name'].apply(lambda x: str(uuid.uuid4()) if not isinstance(x, str) and math.isnan(x) else x)
    # we generate mol from smiles string. we also strip any endline characters and spaces
    smis['mol'] = smis['smiles'].apply(lambda x: Chem.MolFromSmiles(x.strip()))

fastas = pd.DataFrame()
if len(fasta_strings) > 0:
    # we split the string by >
    fasta_list = filter(None, fasta_strings.split('>'))
    # we remove any redundant newline characters (only one should remain at the center of the text)
    stripped = [x.strip('\n') for x in fasta_list]
    # we split by newline to [name, fasta_sequence]
    groupped = [x.split('\n') for x in stripped]
    # Special case for empty name
    groupped = [[str(uuid.uuid4()), x[0]] if len(x) < 2 else x for x in groupped]
    # We load it into a dataframe with the same dimensions as the smiles dataframe
    fastas = pd.DataFrame(groupped, columns=['name', 'sequence'])
    # We concert fasta -> smiles -> mol
    mols = list()
    for x in fastas['sequence']:
        try:
            mol = Chem.MolFromFASTA(x.strip())
        except:
            st.error(f'Peptide with sequence {x} can not be converted into SMILES. It is not added to the prediction list.')
        mols.append(mol)
    fastas['mol'] = mols

# We combine the two dataframes and reset the index
combined_df = pd.concat((smis, fastas))
combined_df = combined_df.reset_index()

allowed = True
if combined_df.shape[0] > 100:
    allowed = False
    st.error(f'To serve everyone in the CPP research community, we have to limit the number of peptides per run to 100. You supplied {combined_df.shape[0]} peptides.')

if combined_df.size > 0 and allowed:
    st.markdown("""---""")

    submit = st.button("Predict")
    if submit:
        # Featurizing input molecules
        feature_progressbar = st.progress(0, text="Featurizing input molecules...")
        graphs = list()
        for index, row in combined_df.iterrows():
            if row['mol'] is not None:
                data = _featurize_mol(row['mol'])
                if hparams['fingerprint_type'] is not None:
                    fp = fp_dict[hparams['fingerprint_type']].GetFingerprint(row['mol'])
                    data.fp = torch.tensor([fp], dtype=torch.float32)
                graphs.append((row['name'], data))
            feature_progressbar.progress(index/len(combined_df), text=f"Featurizing {row['name']}.")
        feature_progressbar.empty()

        # Making the predictions
        with st.spinner(text="Fetching model prediction..."):
            with torch.no_grad():
                for graph_list in graphs:
                    prediction = model(graph_list[1])[0] # The 1th index is the mol object
                    prediction_float = prediction.item() # We get the item from the tensor
                    # NOTE: we apply a sigmoid at the end just to have the values between 0-1
                    predictions.append((graph_list[0], torch.sigmoid(prediction).item(), 'Yes' if prediction_float >= 0.5 else 'No'))

        # Display the predictions
        st.write('## Predictions')
        df = pd.DataFrame(predictions, columns=['Name', 'Probability', 'Cell-Penetrating'])
        df.index.name = '#'

        # Display the dataframe
        st.dataframe(df, use_container_width=True)

        # Create download button
        st.download_button(
            "Download prediction(s) as csv",
            df.to_csv().encode('utf-8'),
            "prediction.csv",
            "text/csv",
            key='download-csv'
        )

# Other misc things
st.markdown("---")
st.markdown("**Reference**: Imre, A., Balogh, B., & Mándity, I. (2024). GraphCPP: The new state-of-the-art method for cell-penetrating peptide prediction via graph neural networks. British Journal of Pharmacology, 1-15. https://doi.org/10. 1111/bph.17388")