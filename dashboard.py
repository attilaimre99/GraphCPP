import streamlit as st
import pandas as pd
import torch
from io import StringIO
from itertools import groupby

from graphcpp.dataset import _featurize_mol
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from graphcpp.lightning import GraphCPPModule

def fasta_iter(string):
    faiter = (x[1] for x in groupby(string, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)

# ----------- General things
col1, mid, col2 = st.columns([1,1,15])
with col1:
    st.image('assets/logo.png', width=100)
with col2:
    st.title('GraphCPP')
# st.title('GraphCPP')
st.subheader('The state-of-the-art graph neural network for the prediction of cell-penetrating peptides.')

loaded_molecules = list()
selection = None
submit = None
st.caption("Made by [Attila Imre](https://github.com/attilaimre99), Balázs Balogh PhD, István Mándity PhD")
st.write("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sed libero dolor. Pellentesque non tempor sem. Aliquam suscipit varius posuere. Integer condimentum a odio sit amet fermentum. Pellentesque eget dapibus tellus. Donec luctus lacinia rutrum. Praesent leo ligula, tristique non rutrum et, maximus sagittis velit. Sed eget scelerisque nisl, non malesuada lectus. Mauris in vulputate neque. Ut vitae porttitor augue. Praesent et tellus sed sem luctus tristique. Fusce lacus magna, consequat sed mattis ac, tempus vel orci. Sed et arcu non lorem ullamcorper venenatis. Integer a urna erat.
""")

smiles_strings = st.text_area('Prediction on SMILES in csv format. If name is not supplied a random name will be generated.', help='Enter valid SMILES strings.', placeholder='name,smiles\nfirst,CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCSC)NC(=O)[C@H](CS)NC(=O)[C@H](CCCCN)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]cn1)NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@@H](N)CCCCN)[C@@H](C)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O')
fasta_strings = st.text_area('Prediction on FASTA. If name is not supplied a random name will be generated.', help='Enter valid FASTA.', placeholder='>First\nRRRRRRRR\n>Second\nWWWWWWWW')

smis = pd.DataFrame()
fastas = pd.DataFrame()
if len(smiles_strings) > 0:
    smis = pd.read_csv(StringIO(smiles_strings))
    smis['mol'] = smis['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
if len(fasta_strings) > 0:
    fasta_list = filter(None, fasta_strings.split('>'))
    stripped = [x.strip('\n') for x in fasta_list]
    groupped = [x.split('\n') for x in stripped]
    fastas = pd.DataFrame(groupped, columns=['name', 'smiles'])
    fastas['mol'] = fastas['smiles'].apply(lambda x: Chem.MolFromFASTA(x))
    print(fastas)

combined_df = pd.concat((smis, fastas))
combined_df = combined_df.reset_index()

if not combined_df.size > 0:
  st.warning('Please enter valid sequences.')

if combined_df.size > 0:
    st.markdown("""---""")

    submit = st.button("Predict")
    if submit:
        with st.spinner(text="Featurizing input molecules..."):
            graphs = list()
            for index, row in combined_df.iterrows():
                if row['mol'] is not None:
                    graphs.append((row['name'], _featurize_mol(row['mol'])))
                    print(row['name'], row['mol'])
            
        with st.spinner(text="Fetching model prediction..."):
            model = GraphCPPModule.load_from_checkpoint(checkpoint_path="model/epoch=38-step=7020.ckpt")
            model.eval()
            predictions = list()
            for graph_list in graphs:
                
                print(graph_list)
                with torch.no_grad():
                    prediction = model(graph_list[1])[0]
                    prediction_float = prediction.item()
                    predictions.append((graph_list[0], torch.sigmoid(prediction).item(), 'Yes' if prediction_float >= 0.5 else 'No'))

        st.write('## Predictions')
        df = pd.DataFrame(predictions, columns=['Name', 'Probability', 'Cell-Penetrating'])
        df.index.name = '#'
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download prediction(s) as csv",
            df.to_csv().encode('utf-8'),
            "prediction.csv",
            "text/csv",
            key='download-csv'
        )

st.markdown("---")
st.markdown("**Reference**: TODO")