import streamlit as st
import pandas as pd
import torch
import uuid
import math
from graphcpp.lightning import GraphCPPModule
from graphcpp.dataset import _featurize_mol
from rdkit import Chem
from io import StringIO

# ----------- Default values
submit = None

# ----------- General things
col1, mid, col2 = st.columns([1,1,15])
with col1:
    st.image('assets/logo.png', width=100)
with col2:
    st.title('GraphCPP')
st.subheader('The state-of-the-art graph neural network for the prediction of cell-penetrating peptides.')
st.caption("Made by Attila Imre ([github](https://github.com/attilaimre99)), Balázs Balogh PhD ([orcid](https://orcid.org/0000-0001-7282-7283)), István Mándity PhD ([orcid](https://orcid.org/0000-0003-2865-6143))")
st.write("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sed libero dolor. Pellentesque non tempor sem. Aliquam suscipit varius posuere. Integer condimentum a odio sit amet fermentum. Pellentesque eget dapibus tellus. Donec luctus lacinia rutrum. Praesent leo ligula, tristique non rutrum et, maximus sagittis velit. Sed eget scelerisque nisl, non malesuada lectus. Mauris in vulputate neque. Ut vitae porttitor augue. Praesent et tellus sed sem luctus tristique. Fusce lacus magna, consequat sed mattis ac, tempus vel orci. Sed et arcu non lorem ullamcorper venenatis. Integer a urna erat.
""")

# ----------- Input fields
smiles_strings = st.text_area('Prediction on SMILES in csv format. If name is not supplied a random name will be generated.', help='Enter valid SMILES strings.', placeholder='name,smiles\nfirst,CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCSC)NC(=O)[C@H](CS)NC(=O)[C@H](CCCCN)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]cn1)NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@@H](N)CCCCN)[C@@H](C)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O')
fasta_strings = st.text_area('Prediction on FASTA. If name is not supplied a random name will be generated.', help='Enter valid FASTA.', placeholder='>First\nRRRRRRRR\n>Second\nWWWWWWWW')

# ----------- SMILES processing
smis = pd.DataFrame()
if len(smiles_strings) > 0:
    # we read in the text as it was a csv file without header
    smis = pd.read_csv(StringIO(smiles_strings), header=None, names=['name', 'sequence'])
    # we generate uuid4 names for empty name fields
    smis['name'] = smis['name'].apply(lambda x: str(uuid.uuid4()) if not isinstance(x, str) and math.isnan(x) else x)
    # we generate mol from smiles string. we also strip any endline characters and spaces
    smis['mol'] = smis['sequence'].apply(lambda x: Chem.MolFromSmiles(x.strip()))

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
    fastas['mol'] = fastas['sequence'].apply(lambda x: Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromFASTA(x.strip()))))

# We combine the two dataframes and reset the index
combined_df = pd.concat((smis, fastas))
combined_df = combined_df.reset_index()

if combined_df.size > 0:
    st.markdown("""---""")

    submit = st.button("Predict")
    if submit:
        # Featurizing input molecules
        feature_progressbar = st.progress(0, text="Featurizing input molecules...")
        graphs = list()
        for index, row in combined_df.iterrows():
            if row['mol'] is not None:
                graphs.append((row['name'], _featurize_mol(row['mol'])))
            feature_progressbar.progress(index/len(combined_df), text=f"Featurizing {row['name']}.")
        feature_progressbar.empty()

        # Making the predictions
        with st.spinner(text="Fetching model prediction..."):
            # We load in the best model from the model path
            model = GraphCPPModule.load_from_checkpoint(checkpoint_path="model/epoch=38-step=7020.ckpt")
            model.eval()
            # New list to hold the predictions
            predictions = list()
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
st.markdown("**Reference**: TODO")