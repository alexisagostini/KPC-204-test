# KPC-204-test

## download the KPC-2 cristallography
The cristallography is a really good start, we have the exact folding of the protein KPC-2 
```bash
wget https://files.rcsb.org/download/2OV5.pdb
mv 2OV5.pb KPC-2cristalo
```
## Make a KPC-204 .pdb from swiss model.
I don't have cristallography for the KPC-204, i will try to manage with swissmodel in order to have a folding based on the KPC-2 folding and the 3 amino acide include in the KPC-204.
I primarly need a fasta of the protein KPC-204
```bash
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=WXU16489.1&rettype=fasta&retmode=text" -O KPC204.fasta
```
On the swiss model web site (https://swissmodel.expasy.org/interactive#sequence) works on the 28/04/2026 add the fasta file and download the .pbd
rename to KCP-204swissmodel.pdb

## Alpha fold
in order to verifie if swiss model made a good folding, I will fold both protein from the scratch.
if the folding of KPC-2 is similare to the cristallography the folding by alphafold will be a good model for KPC-204 otherwise using swissmodel could be the best way.

### downloading micromamba
This micro-environment is a good way to download alpha fold 
```bash 
micromamba create -n colabfold python=3.10 -c conda-forge -y
micromamba activate colabfold
```
and download some tools usefull
```bash 
micromamba install -c bioconda hhsuite -y
pip install pdbfixer
```
### Downloading ColabFold
AlphaFold is thinked for thousand protein and the programme is really heavy and complex, ColabFold is lighter and faster especially if we work on little known protein like i do
```bash
pip install \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

pip install "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
I also need to install the weights of the AlphaFold2 neural network.
```python
#python
python -c "from colabfold.download import download_alphafold_params; download_alphafold_params('/data/alexis/colabfold_params')"
```
### launch the prediction 
```bash 
cat > /data/alexis/project/run_colabfold.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=colabfold_KPC204
#SBATCH --output=/data/alexis/project/KPC204/logs/colabfold_%j.log
#SBATCH --error=/data/alexis/project/KPC204/logs/colabfold_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

eval "$(micromamba shell hook --shell bash)"
micromamba activate /data/alexis/envs/colabfold

colabfold_batch \
    /data/alexis/project/KPC204.fasta \
    /data/alexis/project/KPC204/output \
    --data /data/alexis/colabfold_params \
    --num-recycle 3 \
    --model-type alphafold2_ptm \
    --templates \
    --amber
EOF

sbatch /data/alexis/project/run_colabfold.sh
```
JAX could not detect GPU if is version is too old, feel free to update it

Copy the file and move it with an easier name
```bash 
cp /data/alexis/project/KPC204/output/WXU16489.1_inhibitor-resistant_carbapenem-hydrolyzing_class_A_beta-lactamase_KPC-204__plasmid___Klebsiella_pneumoniae__relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb \
   /data/alexis/project/KPC204/KCPpdb/KPC204alphafold.pdb
```
same for the KPC-2alphafold
```bash
wget -O KPC-2.fasta "https://rest.uniprot.org/uniprotkb/Q9F663.fasta"
```
I have on my repository
```
KPCpdb
|--KCP-2AlphaFold.pdb
|--KCP-2cristalo.pdb
|--KCP-204swissmodel.pdb
|--KPC204alphafold.pdb
```
I need a comparaison point to validate my KCP proteines's fold to compare i need the tools tmalign
```bash
conda install -c bioconda tmalign
```
In order to compare the structure of protein i use that code 
```bash
TMalign Prot1.pdb Prot2.pdb
```
I have several parameters and I accept a mean deviation of 2 amstrong (RMSD=2) equivalent at a covalent bind and a normality score of 0,70(TMscore=0,70) (Yang Zhang and Jeffrey Skolnick 2004 found that 0,47 was a good score) and an identity of sequence of 0,9 id_seq= 0,9

|Protein 1|Protein 2| TM score | RMSD | seq_ID |word|
|---|---|---|---|---|---|
|KPC-2cristalo_clean.pdb|KCP-2cristalo.pdb|1.000|0.000|1.000|Identical|
|KPC-2cristalo_clean.pdb|KCP-2alphafold.pdb|0.88681|0.42|1.000|really similare protein from alphafold seems to be a good model|
|KPC-204swissmodel.pdb|KCP-204alphafold.pdb|0.89864|1.21|0,978|swiss model looks loke a good model but different from alpha fold|
|KCP-2alphafold.pdb|KCP-204alphafold.pdb|0.94515|1.14|0.993|The comparaison between both alphafold protein looks good the difference probably comes from the 3 amino acide|
|KCP-2alphafold.pdb|KCP-204swissmodel.pdb|0.90273|1.10|0.974|The comparaison between both  protein looks good the difference probably comes from the 3 amino acide|

All protein seems to have really well folded i just decide to exlude KPC-2cristalo.pdb because I have a similar one but cleanner



